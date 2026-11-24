
import uuid

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


# ============================================================
# 1. VIDEOS
# ============================================================

class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    source_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,
    )
    video_title: Mapped[str | None] = mapped_column(String(500))
    media_type: Mapped[str] = mapped_column(
        String(30), default="video"
    )
    language: Mapped[str | None] = mapped_column(String(20))

    speaker_name: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    speaker_profession: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    speaker_qualification: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    creator_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    creator_type_other: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    processing_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        index=True,
    )
    failure_stage: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        uselist=False,
    )

    segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
    )

    errors: Mapped[list["AnalysisError"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
    )


# ============================================================
# 2. TRANSCRIPTS
# ============================================================

class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    video_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(20))
    total_segments: Mapped[int] = mapped_column(
        Integer, default=0
    )

    raw_transcript: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    video: Mapped["Video"] = relationship(
        back_populates="transcript"
    )


# ============================================================
# 3. TRANSCRIPT SEGMENTS
# ============================================================

class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    video_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )

    segment_index: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    category: Mapped[str | None] = mapped_column(String(100))
    reason: Mapped[str | None] = mapped_column(Text)

    requires_fact_check: Mapped[bool | None] = mapped_column(Boolean)
    classification_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    needs_review: Mapped[bool | None] = mapped_column(Boolean)
    timing: Mapped[str | None] = mapped_column(String(50))

    raw_ai_output: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    video: Mapped["Video"] = relationship(
        back_populates="segments"
    )

    claims: Mapped[list["Claim"]] = relationship(
        back_populates="segment",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_segments_video_index", "video_id", "segment_index"),
    )


# ============================================================
# 4. CLAIMS
# ============================================================

class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE"),
        nullable=False,
    )

    claim_text: Mapped[str] = mapped_column(Text, nullable=False)

    claim_type: Mapped[str | None] = mapped_column(String(100))
    risk: Mapped[str | None] = mapped_column(String(50))
    hedged: Mapped[bool | None] = mapped_column(Boolean)
    absolute_wording: Mapped[bool | None] = mapped_column(Boolean)
    verdict: Mapped[str | None] = mapped_column(String(100))

    raw_ai_output: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    segment: Mapped["TranscriptSegment"] = relationship(
        back_populates="claims"
    )

    search_queries: Mapped[list["ClaimSearchQuery"]] = relationship(
        back_populates="claim",
        cascade="all, delete-orphan",
    )

    reviews: Mapped[list["ClaimReview"]] = relationship(
        back_populates="claim",
        cascade="all, delete-orphan",
    )


# ============================================================
# 5. CLAIM SEARCH QUERIES
# ============================================================

class ClaimSearchQuery(Base):
    __tablename__ = "claim_search_queries"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
    )

    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_language: Mapped[str | None] = mapped_column(String(20))

    claim: Mapped["Claim"] = relationship(
        back_populates="search_queries"
    )


# ============================================================
# 6. ANALYSIS ERRORS
# ============================================================

class AnalysisError(Base):
    __tablename__ = "analysis_errors"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    video_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )

    stage: Mapped[str] = mapped_column(String(100), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    video: Mapped["Video"] = relationship(
        back_populates="errors"
    )


# ============================================================
# 7. HUMAN CLAIM REVIEWS
# ============================================================


# ============================================================
# 7. HUMAN CLAIM REVIEWS
# ============================================================

class ClaimReview(Base):
    __tablename__ = "claim_reviews"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
    )

    reviewer_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # Q1: Initial human review
    reviewed_label: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    corrected_claim_text: Mapped[str | None] = mapped_column(Text)
    review_notes: Mapped[str | None] = mapped_column(Text)

    # Q2: Claim truth verification
    truth_status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Q3: Source verification
    source_category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    source_name: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    source_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    evidence_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Review completion
    is_final: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    claim: Mapped["Claim"] = relationship(
        back_populates="reviews"
    )

