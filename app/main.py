import logging
import uuid
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import (
    Video,
    Transcript,
    TranscriptSegment as TranscriptSegmentDB,
    Claim,
    ClaimSearchQuery,
    ClaimReview,
)
from app.services.claim_analysis import analyze_segments
from app.services.claim_normalization import normalize_claim
from app.services.evidence_retrieval import (
    normalize_evidence_url,
    retrieve_evidence_for_claim,
)
from app.services.evidence_stance import classify_evidence_passages
from app.services.source_verification import verify_evidence
from app.services.url_processing import process_video_url

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str

class EvidenceStanceRequest(BaseModel):
    claim: str
    evidence: list[str]


class TranscriptAnalysisRequest(BaseModel):
    segments: list[TranscriptSegment]


class VideoURLRequest(BaseModel):
    url: HttpUrl
    mode: Literal["auto", "urdu", "english", "roman_urdu"] = "auto"

    # Optional speaker information
    speaker_name: str | None = None
    speaker_profession: str | None = None
    speaker_qualification: str | None = None
    creator_type: str | None = None
    creator_type_other: str | None = None



# ============================================================
# HUMAN REVIEW REQUEST SCHEMAS
# ============================================================


class ClaimReviewDecision(BaseModel):
    claim_id: uuid.UUID

    # Q1: Initial human review
    reviewed_label: Literal[
        "agree",
        "disagree",
        "neutral",
        "not_a_claim",
    ]

    # Existing review fields
    review_notes: str | None = None
    corrected_claim_text: str | None = None

    # Q2: Truth verification (only after Agree)
    truth_status: Literal[
                      "accurate",
                      "misleading",
                      "false",
                  ] | None = None

    # Q3: Source category
    source_category: Literal[
        "Pakistan Dietary Guidelines for Better Nutrition (2018)",
        "WHO Guidelines",
        "Disease Guidelines",
        "Supplement and Health Claims References",
        "Systematic Reviews and Meta-Analyses",
        "Other",
        "Cannot be verified",
    ] | None = None

    # Actual source document or organization
    source_name: str | None = None

    # Actual evidence webpage
    source_url: HttpUrl | None = None

    # Explanation supporting the truth status
    evidence_text: str | None = None


class SubmitClaimReviewsRequest(BaseModel):
    video_id: uuid.UUID
    reviews: list[ClaimReviewDecision]



class SubmitReviewRequest(BaseModel):
    reviewer_id: str
    decisions: list[ClaimReviewDecision]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VerityVision AI",
    description="Evidence-based video fact-checking and transcription API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "project": "ClaimCheck Video AI",
        "message": "Welcome to ClaimCheck Video AI",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "project": "ClaimCheck Video AI",
    }


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@app.post("/transcribe/url")
async def transcribe_video_url(request: VideoURLRequest):
    try:
        result = await run_in_threadpool(
            process_video_url,
            str(request.url),
            request.mode,
        )

        return {
            "status": "success",
            "filename": result["filename"],
            "media_type": "video",
            "transcript": result["transcript"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Video transcription failed")

        raise HTTPException(
            status_code=500,
            detail="Video transcription failed. Check the server logs.",
        ) from exc





# ---------------------------------------------------------------------------
# Video transcription + claim extraction only
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Video transcription + claim extraction + PostgreSQL storage
# ---------------------------------------------------------------------------



@app.post("/analyze/video-to-claims")
async def analyze_video_to_claims(
    request: VideoURLRequest,
    db: Session = Depends(get_db),
):
    video = None
    current_stage = "video_creation"

    try:
        # --------------------------------------------------
        # STAGE 0: Prevent duplicate videos
        # --------------------------------------------------

        source_url = str(request.url).rstrip("/")

        existing_video = (
            db.query(Video)
            .filter(Video.source_url == source_url)
            .first()
        )

        if existing_video is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This video has already been submitted.",
                    "video_id": str(existing_video.id),
                    "video_title": existing_video.video_title,
                    "processing_status": existing_video.processing_status,
                },
            )

        # --------------------------------------------------
        # STAGE 0A: Create new video record
        # --------------------------------------------------

        video = Video(
            source_url=source_url,

            # Optional speaker information
            speaker_name=(
                request.speaker_name.strip()
                if request.speaker_name and request.speaker_name.strip()
                else None
            ),
            speaker_profession=(
                request.speaker_profession.strip()
                if request.speaker_profession and request.speaker_profession.strip()
                else None
            ),
            speaker_qualification=(
                request.speaker_qualification.strip()
                if request.speaker_qualification and request.speaker_qualification.strip()
                else None
            ),
            # Creator type information
            creator_type=(
                request.creator_type.strip()
                if request.creator_type and request.creator_type.strip()
                else None
            ),
            creator_type_other=(
                request.creator_type_other.strip()
                if request.creator_type_other
                   and request.creator_type_other.strip()
                else None
            ),

            processing_status="processing",
        )

        db.add(video)

        try:
            db.commit()

        except IntegrityError:
            db.rollback()

            # Another request may have submitted the same URL
            existing_video = (
                db.query(Video)
                .filter(Video.source_url == source_url)
                .first()
            )

            if existing_video is not None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "This video has already been submitted.",
                        "video_id": str(existing_video.id),
                        "video_title": existing_video.video_title,
                        "processing_status": existing_video.processing_status,
                    },
                )

            raise

        db.refresh(video)

        # --------------------------------------------------
        # STAGE 1: Download video and generate transcript
        # --------------------------------------------------

        current_stage = "transcription"

        result = await run_in_threadpool(
            process_video_url,
            source_url,
            request.mode,
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Video processing returned an invalid result."
            )

        transcript = result.get("transcript")

        if not isinstance(transcript, dict):
            raise ValueError(
                "Video transcription returned an invalid result."
            )

        transcript_segments = transcript.get("segments", [])

        if not isinstance(transcript_segments, list):
            raise ValueError(
                "Transcript segments have an invalid format."
            )

        transcript_text = transcript.get("text", "")
        transcript_language = transcript.get("language", "unknown")

        if not isinstance(transcript_text, str):
            raise ValueError(
                "Transcript text has an invalid format."
            )

        # --------------------------------------------------
        # STAGE 1A: Update video metadata
        # --------------------------------------------------

        current_stage = "video_metadata"

        video.video_title = (
            result.get("video_title")
            or result.get("filename")
            or "Untitled video"
        )

        video.media_type = "video"
        video.language = transcript_language
        video.processing_status = "transcribed"

        db.commit()

        # --------------------------------------------------
        # STAGE 1B: Save complete transcript
        # --------------------------------------------------

        current_stage = "transcript_saving"

        transcript_record = Transcript(
            video_id=video.id,
            full_text=transcript_text,
            language=transcript_language,
            total_segments=len(transcript_segments),
            raw_transcript=transcript,
        )

        db.add(transcript_record)
        db.commit()
        db.refresh(transcript_record)

        # --------------------------------------------------
        # STAGE 2: Classify segments and extract claims
        # --------------------------------------------------

        current_stage = "claim_analysis"

        analyzed_segments = await run_in_threadpool(
            analyze_segments,
            transcript_segments,
        )

        if not isinstance(analyzed_segments, list):
            raise ValueError(
                "Claim analysis returned an invalid result."
            )

        # --------------------------------------------------
        # STAGE 3: Detect analysis failures
        # --------------------------------------------------

        analysis_failed = any(
            isinstance(segment, dict)
            and (
                segment.get("_classification_failed", False)
                or segment.get("_claim_extraction_failed", False)
            )
            for segment in analyzed_segments
        )

        # --------------------------------------------------
        # STAGE 4: Save segments, claims and search queries
        # --------------------------------------------------

        current_stage = "database_saving"

        for index, segment in enumerate(analyzed_segments):

            if not isinstance(segment, dict):
                logger.warning(
                    "Skipping invalid segment at index %s",
                    index,
                )
                continue

            segment_text = segment.get("text", "")

            if not isinstance(segment_text, str):
                segment_text = str(segment_text)

            try:
                start_time = float(segment.get("start", 0))
                end_time = float(segment.get("end", 0))

            except (TypeError, ValueError):
                logger.warning(
                    "Skipping segment with invalid timestamps: %s",
                    index,
                )
                continue

            # Save transcript segment.
            segment_record = TranscriptSegmentDB(
                video_id=video.id,
                segment_index=index,
                start_time=start_time,
                end_time=end_time,
                text=segment_text,
                category=segment.get("category"),
                reason=segment.get("reason"),
                requires_fact_check=segment.get(
                    "requires_fact_check"
                ),
                classification_confidence=segment.get(
                    "classification_confidence"
                ),
                needs_review=segment.get("needs_review"),
                timing=segment.get("timing"),
                raw_ai_output=segment,
            )

            db.add(segment_record)
            db.flush()

            # --------------------------------------------------
            # STAGE 4A: Save claims
            # --------------------------------------------------

            claims = segment.get("claims", [])

            if not isinstance(claims, list):
                continue

            for claim_data in claims:

                if not isinstance(claim_data, dict):
                    continue

                claim_text = claim_data.get("text")

                if (
                    not isinstance(claim_text, str)
                    or not claim_text.strip()
                ):
                    continue

                claim_record = Claim(
                    segment_id=segment_record.id,
                    claim_text=claim_text.strip(),
                    claim_type=claim_data.get("claim_type"),
                    risk=claim_data.get("risk"),
                    hedged=claim_data.get("hedged"),
                    absolute_wording=claim_data.get(
                        "absolute_wording"
                    ),
                    verdict=claim_data.get("verdict"),
                    raw_ai_output=claim_data,
                )

                db.add(claim_record)
                db.flush()

                claim_data["_database_id"] = str(claim_record.id)

                # --------------------------------------------------
                # STAGE 4B: Save claim search queries
                # --------------------------------------------------

                search_queries = claim_data.get(
                    "search_queries",
                    [],
                )

                if not isinstance(search_queries, list):
                    continue

                for query in search_queries:

                    if (
                        not isinstance(query, str)
                        or not query.strip()
                    ):
                        continue

                    query_language = (
                        "ur"
                        if any(
                            "\u0600" <= char <= "\u06FF"
                            for char in query
                        )
                        else "en"
                    )

                    query_record = ClaimSearchQuery(
                        claim_id=claim_record.id,
                        query_text=query.strip(),
                        query_language=query_language,
                    )

                    db.add(query_record)

        # Save all segments, claims and queries.
        db.commit()

        # --------------------------------------------------
        # STAGE 5: Prepare public API response
        # --------------------------------------------------

        current_stage = "response_preparation"

        excluded_claim_fields = {
            "risk",
            "hedged",
            "absolute_wording",
            "verdict",
            "claim_type",
        }

        public_segments = []

        for segment in analyzed_segments:

            if not isinstance(segment, dict):
                continue

            public_segment = {
                key: value
                for key, value in segment.items()
                if not key.startswith("_")
            }

            claims = public_segment.get("claims", [])

            if not isinstance(claims, list):
                claims = []

            public_claims = []

            for claim in claims:

                if not isinstance(claim, dict):
                    continue

                public_claim = {
                    key: value
                    for key, value in claim.items()
                    if (
                        not key.startswith("_")
                        and key not in excluded_claim_fields
                    )
                }

                if claim.get("_database_id"):
                    public_claim["claim_id"] = claim["_database_id"]

                public_claims.append(public_claim)

            public_segment["claims"] = public_claims

            public_segments.append(public_segment)

        # --------------------------------------------------
        # STAGE 6: Update final processing status
        # --------------------------------------------------

        current_stage = "final_status_update"

        video.processing_status = (
            "partial_success"
            if analysis_failed
            else "completed"
        )

        db.commit()

        # --------------------------------------------------
        # STAGE 7: Return response
        # --------------------------------------------------

        return {
            "status": (
                "partial_success"
                if analysis_failed
                else "success"
            ),
            "video_id": str(video.id),
            "video_title": video.video_title,
            "media_type": video.media_type,
            "language": video.language,
            "text": transcript_text,
            "analysis": {
                "total_segments": len(public_segments),
                "segments": public_segments,
            },
        }

    # ------------------------------------------------------
    # ERROR HANDLING
    # ------------------------------------------------------

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:

        logger.warning(
            "Video processing validation error at stage %s: %s",
            current_stage,
            str(exc),
        )

        db.rollback()

        if video is not None:

            try:
                video = db.get(Video, video.id)

                if video is not None:
                    video.processing_status = "failed"
                    video.failure_stage = current_stage
                    db.commit()

            except Exception:
                db.rollback()

                logger.exception(
                    "Could not update video failure status."
                )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Video-to-claims analysis failed at stage %s",
            current_stage,
        )

        db.rollback()

        if video is not None:

            try:
                video = db.get(Video, video.id)

                if video is not None:
                    video.processing_status = "failed"
                    video.failure_stage = current_stage
                    db.commit()

            except Exception:
                db.rollback()

                logger.exception(
                    "Could not update video failure status."
                )

        raise HTTPException(
            status_code=500,
            detail=(
                "Video-to-claims analysis failed. "
                "Check server logs."
            ),
        ) from exc



# --------------------------------------------------
# STAGE 11: Submit human claim reviews
# --------------------------------------------------



# --------------------------------------------------
# STAGE 11: Submit human claim reviews
# --------------------------------------------------


@app.post("/reviews/submit")
async def submit_claim_reviews(
    request: SubmitClaimReviewsRequest,
    db: Session = Depends(get_db),
):
    try:
        # --------------------------------------------------
        # 1. Validate review submission
        # --------------------------------------------------

        if not request.reviews:
            raise HTTPException(
                status_code=400,
                detail="At least one claim review is required.",
            )

        # Prevent duplicate claim IDs.
        submitted_claim_ids = [
            review.claim_id for review in request.reviews
        ]

        if len(submitted_claim_ids) != len(set(submitted_claim_ids)):
            raise HTTPException(
                status_code=400,
                detail="Duplicate claim IDs are not allowed.",
            )

        # --------------------------------------------------
        # 2. Validate Q1, Q2 and Q3
        # --------------------------------------------------

        for review in request.reviews:

            if review.reviewed_label == "agree":

                # Q2: Truth status is mandatory.
                if not review.truth_status:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Truth status is required for "
                            f"claim {review.claim_id}."
                        ),
                    )

                # Q3: Source category is mandatory.
                if not review.source_category:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Source category is required for "
                            f"claim {review.claim_id}."
                        ),
                    )

                # Source details are required only when
                # the source category is not Cannot be verified.
                if review.source_category != "Cannot be verified":

                    if (
                        not review.source_name
                        or not review.source_name.strip()
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Source name or document is required for "
                                f"claim {review.claim_id}."
                            ),
                        )

                    if not review.source_url:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Source URL is required for "
                                f"claim {review.claim_id}."
                            ),
                        )

                    if (
                        not review.evidence_text
                        or not review.evidence_text.strip()
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Evidence explanation is required for "
                                f"claim {review.claim_id}."
                            ),
                        )

        # --------------------------------------------------
        # 3. Check that the video exists
        # --------------------------------------------------

        video = db.get(Video, request.video_id)

        if video is None:
            raise HTTPException(
                status_code=404,
                detail="Video not found.",
            )

        # --------------------------------------------------
        # 4. Validate that every claim belongs to this video
        # --------------------------------------------------

        for review in request.reviews:

            claim_record = (
                db.query(Claim)
                .join(
                    TranscriptSegmentDB,
                    Claim.segment_id == TranscriptSegmentDB.id,
                )
                .filter(
                    Claim.id == review.claim_id,
                    TranscriptSegmentDB.video_id == request.video_id,
                )
                .first()
            )

            if claim_record is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Claim {review.claim_id} does not "
                        "belong to the specified video."
                    ),
                )

        # --------------------------------------------------
        # 5. Save all reviews
        # --------------------------------------------------

        saved_reviews = []

        for review in request.reviews:

            # Find an existing anonymous review.
            review_record = (
                db.query(ClaimReview)
                .filter(
                    ClaimReview.claim_id == review.claim_id,
                    ClaimReview.reviewer_id.is_(None),
                )
                .first()
            )

            # Create a new anonymous review if necessary.
            if review_record is None:

                review_record = ClaimReview(
                    claim_id=review.claim_id,
                    reviewer_id=None,
                )

                db.add(review_record)

            # Q1: Save initial review decision.
            review_record.reviewed_label = review.reviewed_label

            review_record.review_notes = review.review_notes

            review_record.corrected_claim_text = (
                review.corrected_claim_text
            )

            # --------------------------------------------------
            # Q2 and Q3: Save evidence only for Agree
            # --------------------------------------------------

            if review.reviewed_label == "agree":

                review_record.truth_status = review.truth_status

                review_record.source_category = (
                    review.source_category
                )

                if review.source_category == "Cannot be verified":

                    # Save the selected category and truth status.
                    # No source details are available.
                    review_record.source_name = None
                    review_record.source_url = None
                    review_record.evidence_text = None

                else:

                    # Save complete source verification details.
                    review_record.source_name = (
                        review.source_name.strip()
                    )

                    review_record.source_url = str(
                        review.source_url
                    )

                    review_record.evidence_text = (
                        review.evidence_text.strip()
                    )

            else:

                # Clear previous evidence if the decision changes.
                review_record.truth_status = None
                review_record.source_category = None
                review_record.source_name = None
                review_record.source_url = None
                review_record.evidence_text = None

            # --------------------------------------------------
            # Review completion
            # --------------------------------------------------

            review_record.is_final = True

            review_record.reviewed_at = datetime.now(timezone.utc)

            db.flush()

            saved_reviews.append({
                "review_id": str(review_record.id),
                "claim_id": str(review.claim_id),
                "reviewed_label": review.reviewed_label,
                "truth_status": review_record.truth_status,
                "source_category": review_record.source_category,
                "source_name": review_record.source_name,
                "source_url": review_record.source_url,
                "evidence_text": review_record.evidence_text,
            })

        # --------------------------------------------------
        # 6. Commit all reviews together
        # --------------------------------------------------

        db.commit()

        return {
            "status": "success",
            "message": "Claim reviews saved successfully.",
            "video_id": str(request.video_id),
            "total_reviews": len(saved_reviews),
            "reviews": saved_reviews,
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Claim review submission failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Could not save claim reviews.",
        ) from exc


