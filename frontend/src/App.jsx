
import { useState, useEffect, useMemo, useRef } from "react";
import "./App.css";

import {
  checkBackendHealth,
  processVideo,
  submitClaimReviews,
} from "./api";

// ============================================================
// YOUTUBE IFRAME API
// ============================================================

let youtubeApiPromise;

function loadYouTubeApi() {
  if (window.YT?.Player) {
    return Promise.resolve(window.YT);
  }

  if (!youtubeApiPromise) {
    youtubeApiPromise = new Promise((resolve) => {
      const previousCallback = window.onYouTubeIframeAPIReady;

      window.onYouTubeIframeAPIReady = () => {
        previousCallback?.();
        resolve(window.YT);
      };

      const existingScript = document.querySelector(
        'script[src="https://www.youtube.com/iframe_api"]'
      );

      if (!existingScript) {
        const script = document.createElement("script");
        script.src = "https://www.youtube.com/iframe_api";
        script.async = true;
        document.head.appendChild(script);
      }
    });
  }

  return youtubeApiPromise;
}

// ============================================================
// VIDEO URL HELPERS
// ============================================================

function getVideoDetails(rawUrl) {
  try {
    const url = new URL(rawUrl);
    const host = url.hostname.replace(/^www\./, "");

    // YouTube
    if (
      host === "youtube.com" ||
      host === "m.youtube.com" ||
      host === "youtu.be" ||
      host === "youtube-nocookie.com"
    ) {
      let videoId = "";

      if (host === "youtu.be") {
        videoId = url.pathname.split("/")[1];
      } else {
        videoId =
          url.searchParams.get("v") ||
          url.pathname.match(/\/(?:shorts|embed|live)\/([^/?]+)/)?.[1];
      }

      if (videoId) {
        const origin = encodeURIComponent(window.location.origin);

        return {
          provider: "youtube",
          embedUrl: `https://www.youtube-nocookie.com/embed/${videoId}?autoplay=1&mute=1&playsinline=1&enablejsapi=1&origin=${origin}`,
        };
      }
    }

    // Vimeo
    if (host === "vimeo.com" || host === "player.vimeo.com") {
      const videoId = url.pathname.match(/\/(?:video\/)?(\d+)/)?.[1];

      if (videoId) {
        return {
          provider: "vimeo",
          embedUrl: `https://player.vimeo.com/video/${videoId}?autoplay=1&muted=1`,
        };
      }
    }

    // TikTok
    if (host === "tiktok.com" || host === "vm.tiktok.com") {
      const videoId = url.pathname.match(/\/video\/(\d+)/)?.[1];

      if (videoId) {
        return {
          provider: "tiktok",
          embedUrl: `https://www.tiktok.com/player/v1/${videoId}?autoplay=1`,
        };
      }
    }

    // Instagram
    if (host === "instagram.com") {
      const match = url.pathname.match(/^\/(p|reel|tv)\/([^/]+)/);

      if (match) {
        return {
          provider: "instagram",
          embedUrl: `https://www.instagram.com/${match[1]}/${match[2]}/embed`,
        };
      }
    }

    // Facebook
    if (host === "facebook.com" || host === "m.facebook.com") {
      return {
        provider: "facebook",
        embedUrl: `https://www.facebook.com/plugins/video.php?href=${encodeURIComponent(
          rawUrl
        )}&show_text=false&autoplay=true`,
      };
    }

    return null;
  } catch {
    return null;
  }
}

// ============================================================
// TIME FORMATTER
// ============================================================

function formatTime(seconds) {
  const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds % 60;

  return `${String(minutes).padStart(2, "0")}:${String(
    remainingSeconds
  ).padStart(2, "0")}`;
}

// ============================================================
// SPEAKER DROPDOWN OPTIONS
// ============================================================

const SPEAKER_NAME_OPTIONS = [
  "Dr. Muhammad Affan Qaiser",
  "Dr. Waseem",
  "Dr. Asif Izhar",
  "Dr. Hasan Nasir",
  "Dr. Sharafat Ali",
  "Dr. Muhammad Ibrahim",
  "Dr. Mahboob Ahmad",
  "Dr. Umme Raheel",
  "Dr. Fazeela Abbasi",
  "Dr. Bilquis Shaikh",
  "Dr. Samina Haq",
  "Dr. Zohaib & Dr. Shoaib (The Medbros)",
  "Ayesha Nasir",
  "Shahzad Basra",
  "Hakeem Imran Kamboh",
  "Dr. Usman Riaz",
  "Dr. Eric Berg (Urdu-language videos)",
  "Dr. Imran Raza (USA)",
  "Dr. Jahanzeb / Dr. Zeb (USA)",
];

const SPEAKER_PROFESSION_OPTIONS = [
  "Medical Doctor / Physician",
  "General Physician / Family Medicine Doctor",
  "Medical Specialist",
  "Surgeon",
  "Cardiologist",
  "Neurologist",
  "Neurosurgeon",
  "Endocrinologist / Diabetes Specialist",
  "Gastroenterologist",
  "Pulmonologist / Chest Specialist",
  "Dermatologist / Skin Specialist",
  "Psychiatrist",
  "Clinical Psychologist",
  "Dentist / Dental Surgeon",
  "Nutritionist / Dietitian",
  "Physiotherapist",
  "Pharmacist",
  "Nurse",
  "Public Health Professional",
  "Medical Researcher / Health Scientist",
  "Medical Professor / Health Educator",
  "Homeopathic Practitioner",
  "Unani / Tibb Practitioner (Hakeem)",
  "Health & Wellness Coach",
  "Fitness Trainer / Exercise Specialist",
  "Other Healthcare Professional",
];

const SPEAKER_QUALIFICATION_OPTIONS = [
  "MBBS",
  "MBBS, FCPS",
  "MBBS, MCPS",
  "MBBS, MD",
  "MBBS, MS",
  "MBBS, MRCP (UK)",
  "MBBS, FRCP",
  "MBBS, MPhil",
  "MBBS, PhD",
  "BDS",
  "BDS, FCPS",
  "BDS, MDS",
  "DPT",
  "BS / MSc Physiotherapy",
  "BS Human Nutrition & Dietetics",
  "MSc / MS Nutrition",
  "Pharm-D",
  "BPharm",
  "BS / MSc Nursing",
  "BS / MSc Psychology",
  "MS / MPhil Clinical Psychology",
  "MPH (Master of Public Health)",
  "MSc / MS Public Health",
  "PhD (Health / Medical Sciences)",
  "DHMS (Homeopathy)",
  "BUMS (Unani Medicine)",
  "Diploma in Alternative Medicine",
  "Not provided / Not verified",
];

// ============================================================
// MAIN APP
// ============================================================

function App() {
  // ----------------------------------------------------------
  // VIDEO INPUT
  // ----------------------------------------------------------

  const [videoUrl, setVideoUrl] = useState("");
  const [language, setLanguage] = useState("auto");

  // ----------------------------------------------------------
  // SPEAKER INFORMATION
  // ----------------------------------------------------------

  const [speakerName, setSpeakerName] = useState("");
  const [speakerNameOther, setSpeakerNameOther] = useState("");

  const [speakerProfession, setSpeakerProfession] = useState("");
  const [speakerProfessionOther, setSpeakerProfessionOther] = useState("");

  const [speakerQualification, setSpeakerQualification] = useState("");
  const [speakerQualificationOther, setSpeakerQualificationOther] =
    useState("");

  // ----------------------------------------------------------
  // CREATOR INFORMATION
  // ----------------------------------------------------------

  const [creatorType, setCreatorType] = useState("");
  const [creatorTypeOther, setCreatorTypeOther] = useState("");

  // ----------------------------------------------------------
  // ANALYSIS STATES
  // ----------------------------------------------------------

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // ----------------------------------------------------------
  // VIDEO PLAYER STATES
  // ----------------------------------------------------------

  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [playerStatus, setPlayerStatus] = useState(
    "Loading video player..."
  );

  const [backendStatus, setBackendStatus] = useState("Checking...");

  const iframeRef = useRef(null);
  const playerRef = useRef(null);
  const segmentRefs = useRef([]);

  // ----------------------------------------------------------
  // HUMAN REVIEW STATES
  // ----------------------------------------------------------

  const [reviewIndex, setReviewIndex] = useState(0);
  const [reviewDecisions, setReviewDecisions] = useState({});
  const [reviewSaving, setReviewSaving] = useState(false);
  const [reviewStatus, setReviewStatus] = useState("");
  const [reviewError, setReviewError] = useState("");

  // ----------------------------------------------------------
  // VIDEO DETAILS
  // ----------------------------------------------------------

  const videoDetails = useMemo(
    () => (previewPlaying ? getVideoDetails(videoUrl) : null),
    [previewPlaying, videoUrl]
  );

  // ----------------------------------------------------------
  // TRANSCRIPT SEGMENTS
  // ----------------------------------------------------------

  const segments = result?.analysis?.segments || [];

  const activeSegmentIndex = segments.findIndex((segment, index) => {
    const start = Number(segment.start) || 0;

    const end =
      Number(segment.end) ||
      (index < segments.length - 1
        ? Number(segments[index + 1].start) || start
        : start + 1);

    return playbackTime >= start && playbackTime < end;
  });

  const activeSegment =
    activeSegmentIndex >= 0 ? segments[activeSegmentIndex] : null;

  // ----------------------------------------------------------
  // BACKEND HEALTH CHECK
  // ----------------------------------------------------------

  useEffect(() => {
    checkBackendHealth()
      .then((data) => {
        setBackendStatus(
          data.status === "healthy" ? "Backend Connected" : "Backend Error"
        );
      })
      .catch(() => {
        setBackendStatus("Backend Offline");
      });
  }, []);

  // ----------------------------------------------------------
  // YOUTUBE PLAYER INITIALIZATION
  // ----------------------------------------------------------

  useEffect(() => {
    if (
      !previewPlaying ||
      videoDetails?.provider !== "youtube" ||
      !iframeRef.current
    ) {
      return undefined;
    }

    let cancelled = false;
    let pollInterval;

    loadYouTubeApi().then((YT) => {
      if (cancelled || !iframeRef.current || !YT?.Player) {
        return;
      }

      if (playerRef.current?.destroy) {
        playerRef.current.destroy();
        playerRef.current = null;
      }

      playerRef.current = new YT.Player(iframeRef.current, {
        events: {
          onReady: () => {
            if (!cancelled) {
              setPlayerStatus("YouTube player ready");
            }
          },

          onStateChange: (event) => {
            if (cancelled) return;

            if (event.data === YT.PlayerState.PLAYING) {
              setPlayerStatus("Playing · Transcript sync active");

              if (pollInterval) {
                window.clearInterval(pollInterval);
              }

              pollInterval = window.setInterval(() => {
                const player = playerRef.current;

                if (player?.getCurrentTime) {
                  setPlaybackTime(player.getCurrentTime());
                }
              }, 400);
            } else {
              if (pollInterval) {
                window.clearInterval(pollInterval);
                pollInterval = null;
              }

              const player = playerRef.current;

              if (player?.getCurrentTime) {
                setPlaybackTime(player.getCurrentTime());
              }

              if (event.data === YT.PlayerState.PAUSED) {
                setPlayerStatus("Paused");
              } else if (event.data === YT.PlayerState.ENDED) {
                setPlayerStatus("Video ended");
              }
            }
          },
        },
      });
    });

    return () => {
      cancelled = true;

      if (pollInterval) {
        window.clearInterval(pollInterval);
      }

      if (playerRef.current?.destroy) {
        playerRef.current.destroy();
        playerRef.current = null;
      }
    };
  }, [previewPlaying, videoDetails?.embedUrl, videoDetails?.provider]);

  // ----------------------------------------------------------
  // TRANSCRIPT AUTO SCROLL
  // ----------------------------------------------------------

  useEffect(() => {
    if (activeSegmentIndex >= 0) {
      segmentRefs.current[activeSegmentIndex]?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [activeSegmentIndex]);

  // ----------------------------------------------------------
  // COLLECT DATABASE-BACKED CLAIMS
  // ----------------------------------------------------------

  const allClaims =
    result?.analysis?.segments?.flatMap((segment) =>
      (segment.claims || [])
        .filter((claim) => claim.claim_id)
        .map((claim) => ({
          ...claim,
          segmentStart: segment.start,
          segmentEnd: segment.end,
        }))
    ) || [];

  const currentClaim = allClaims[reviewIndex];

  const currentReview = currentClaim
    ? reviewDecisions[currentClaim.claim_id] || {}
    : {};

  // ----------------------------------------------------------
  // REVIEW OPTIONS
  // ----------------------------------------------------------

  const reviewOptions = [
    {
      value: "agree",
      label: "Agree",
      description: "The claim is correctly extracted.",
    },
    {
      value: "disagree",
      label: "Disagree",
      description: "The claim needs correction.",
    },
    {
      value: "neutral",
      label: "Neutral / Insufficient evidence",
      description: "Unable to make a clear decision.",
    },
    {
      value: "not_a_claim",
      label: "Not a claim",
      description: "This is not a factual claim.",
    },
  ];


const truthStatusOptions = [
  { value: "accurate", label: "Accurate" },
  { value: "misleading", label: "Misleading" },
  { value: "false", label: "False" },
];


  // ==========================================================
  // VIDEO SUBMISSION
  // ==========================================================

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!videoUrl.trim()) {
      alert("Please enter a video URL.");
      return;
    }

    // Resolve speaker name.
    const finalSpeakerName =
      speakerName === "Other"
        ? speakerNameOther.trim()
        : speakerName.trim();

    // Resolve speaker profession.
    const finalSpeakerProfession =
      speakerProfession === "Other"
        ? speakerProfessionOther.trim()
        : speakerProfession.trim();

    // Resolve speaker qualification.
    const finalSpeakerQualification =
      speakerQualification === "Other"
        ? speakerQualificationOther.trim()
        : speakerQualification.trim();

    // Resolve creator type.
    const finalCreatorType = creatorType.trim();

    const finalCreatorTypeOther =
      creatorType === "Other" ? creatorTypeOther.trim() : null;

    // Validate required information.
    if (
      !finalSpeakerName ||
      !finalSpeakerProfession ||
      !finalSpeakerQualification ||
      !finalCreatorType ||
      (creatorType === "Other" && !finalCreatorTypeOther)
    ) {
      alert(
        "Please complete all required speaker and creator information fields."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    setPlaybackTime(0);
    setPlayerStatus("Loading video player...");
    setPreviewPlaying(true);

    setReviewIndex(0);
    setReviewDecisions({});
    setReviewStatus("");
    setReviewError("");

    try {
      const data = await processVideo(videoUrl, language, {
        speaker_name: finalSpeakerName,
        speaker_profession: finalSpeakerProfession,
        speaker_qualification: finalSpeakerQualification,

        // Creator information
        creator_type: finalCreatorType,
        creator_type_other: finalCreatorTypeOther,
      });

      setResult(data);

      console.log("Video analysis result:", data);
    } catch (err) {
      if (err.status === 409) {
        setError(
          err.message || "This video has already been submitted."
        );
      } else {
        setError(err.message || "Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  };

  // ==========================================================
  // VIDEO URL CHANGE
  // ==========================================================

  const handleVideoUrlChange = (value) => {
    setVideoUrl(value);
    setPreviewPlaying(false);
    setPlaybackTime(0);
    setResult(null);
    setError("");
  };

  // ==========================================================
  // TRANSCRIPT SEGMENT CLICK
  // ==========================================================

  const handleSegmentClick = (segment) => {
    const start = Number(segment.start) || 0;

    if (
      videoDetails?.provider === "youtube" &&
      playerRef.current?.seekTo
    ) {
      playerRef.current.seekTo(start, true);
      playerRef.current.playVideo();
      setPlaybackTime(start);
    } else {
      setPlayerStatus(
        "Timestamp seeking is currently available for YouTube videos."
      );
    }
  };

  // ==========================================================
  // HUMAN REVIEW: DECISION CHANGE
  // ==========================================================

  const handleDecision = (value) => {
    if (!currentClaim) return;

    const claimId = currentClaim.claim_id;

    setReviewDecisions((previous) => ({
      ...previous,
      [claimId]: {
        ...previous[claimId],
        reviewed_label: value,

        // Clear evidence fields for non-Agree decisions.
        ...(value !== "agree"
          ? {
              truth_status: null,
              source_category: null,
              source_name: "",
              source_url: "",
              evidence_text: "",
            }
          : {}),
      },
    }));

    setReviewError("");

    // Non-Agree decisions automatically move to the next claim.
    if (value !== "agree" && reviewIndex < allClaims.length - 1) {
      setReviewIndex((previous) => previous + 1);
    }
  };

  // ==========================================================
  // HUMAN REVIEW: FIELD CHANGE
  // ==========================================================

  const handleReviewFieldChange = (field, value) => {
  if (!currentClaim) return;

  setReviewDecisions((previous) => ({
    ...previous,
    [currentClaim.claim_id]: {
      ...previous[currentClaim.claim_id],
      [field]: value,

      // Clear source details when verification is not possible.
      ...(field === "source_category" &&
      value === "Cannot be verified"
        ? {
            source_name: "",
            source_url: "",
            evidence_text: "",
          }
        : {}),
    },
  }));

  setReviewError("");
};

  // ==========================================================
  // HUMAN REVIEW: PREVIOUS CLAIM
  // ==========================================================

  const handlePreviousClaim = () => {
    if (reviewIndex <= 0) return;

    setReviewIndex((previous) => Math.max(0, previous - 1));
    setReviewError("");
  };

  // ==========================================================
  // HUMAN REVIEW: NEXT CLAIM
  // ==========================================================

  const handleNextClaim = () => {
  if (!currentClaim) return;

  const decision = reviewDecisions[currentClaim.claim_id];

  if (!decision?.reviewed_label) {
    setReviewError("Please select a review decision first.");
    return;
  }

  if (decision.reviewed_label === "agree") {
    if (!decision.truth_status) {
      setReviewError("Please select the truth status.");
      return;
    }

    if (!decision.source_category) {
      setReviewError("Please select a source category.");
      return;
    }

    // Source details are not required when verification is impossible.
    if (decision.source_category !== "Cannot be verified") {
      if (
        !decision.source_name?.trim() ||
        !decision.source_url?.trim() ||
        !decision.evidence_text?.trim()
      ) {
        setReviewError(
          "Please complete all mandatory source verification fields."
        );
        return;
      }
    }
  }

  setReviewIndex((previous) =>
    Math.min(allClaims.length - 1, previous + 1)
  );

  setReviewError("");
};

  // ==========================================================
  // HUMAN REVIEW: SUBMIT ALL REVIEWS
  // ==========================================================

  const handleReviewSubmit = async () => {
    if (!result?.video_id) {
      setReviewError("Video ID is missing.");
      return;
    }

    // Validate every claim before submitting.
    for (let index = 0; index < allClaims.length; index++) {
      const claim = allClaims[index];
      const decision = reviewDecisions[claim.claim_id];

      if (!decision?.reviewed_label) {
        setReviewError(
          "Please review every claim before submitting."
        );
        setReviewIndex(index);
        return;
      }

      if (decision.reviewed_label === "agree") {
        if (!decision.truth_status) {
          setReviewError(
            "Please select a truth status for every Agree decision."
          );
          setReviewIndex(index);
          return;
        }

        if (!decision.source_category) {
  setReviewError(
    "Please select a source category for every Agree decision."
  );
  setReviewIndex(index);
  return;
}

if (
  decision.source_category !== "Cannot be verified" &&
  (
    !decision.source_name?.trim() ||
    !decision.source_url?.trim() ||
    !decision.evidence_text?.trim()
  )
) {
  setReviewError(
    "Please complete all source verification fields for every Agree decision."
  );
  setReviewIndex(index);
  return;
}
      }
    }


const reviews = allClaims.map((claim) => {
  const decision = reviewDecisions[claim.claim_id];

  const isAgree = decision.reviewed_label === "agree";

  const cannotBeVerified =
    decision.source_category === "Cannot be verified";

  const truthStatus =
    decision.truth_status === "true"
      ? "accurate"
      : decision.truth_status;

  return {
    claim_id: claim.claim_id,
    reviewed_label: decision.reviewed_label,

    review_notes: "",
    corrected_claim_text: "",

    truth_status: isAgree ? truthStatus : null,

    source_category: isAgree
      ? decision.source_category
      : null,

    source_name:
      isAgree && !cannotBeVerified
        ? decision.source_name.trim()
        : null,

    source_url:
      isAgree && !cannotBeVerified
        ? decision.source_url.trim()
        : null,

    evidence_text:
      isAgree && !cannotBeVerified
        ? decision.evidence_text.trim()
        : null,
  };
});

    setReviewSaving(true);
    setReviewError("");
    setReviewStatus("");

    try {
      const data = await submitClaimReviews(
        result.video_id,
        reviews
      );

      setReviewStatus(
        `Successfully saved ${data.total_reviews} reviews to PostgreSQL.`
      );

      console.log("Review submission result:", data);
    } catch (err) {
      setReviewError(
        err.message || "Failed to save reviews."
      );
    } finally {
      setReviewSaving(false);
    }
  };

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app">
      {/* ======================================================
          NAVBAR
      ====================================================== */}

      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">CC</div>

          <div>
            <h2>
              ClaimCheck <span>AI</span>
            </h2>

            <p>Video Intelligence Platform</p>
          </div>
        </div>

        <div className="nav-status">
          <span className="status-dot"></span>
          {backendStatus}
        </div>
      </header>

      <main className="main-content">
        {/* ====================================================
            WELCOME SECTION
        ==================================================== */}

        <section className="welcome-section">
          <span className="eyebrow">
            AI-POWERED VIDEO ANALYSIS
          </span>

          <h1>
            Understand what's being
            <br />
            <span>claimed in any video.</span>
          </h1>

          <p>
            Transcribe videos, extract factual claims, and collect
            human-reviewed annotations for AI research.
          </p>
        </section>

        {/* ====================================================
            VIDEO UPLOAD FORM
        ==================================================== */}

        <section className="upload-card">
          <div className="card-heading">
            <div>
              <h2>Start a new analysis</h2>
              <p>Enter a supported video URL to begin.</p>
            </div>

            <div className="video-icon">▶</div>
          </div>

          <form onSubmit={handleSubmit}>
            {/* VIDEO URL */}

            <label htmlFor="videoUrl">Video URL</label>

            <input
              id="videoUrl"
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={videoUrl}
              onChange={(event) =>
                handleVideoUrlChange(event.target.value)
              }
              required
            />

            {/* LANGUAGE */}

            <label htmlFor="language">
              Transcript language
            </label>

            <select
              id="language"
              value={language}
              onChange={(event) =>
                setLanguage(event.target.value)
              }
            >
              <option value="auto">Auto detect</option>
              <option value="urdu">Urdu</option>
              <option value="english">English</option>
              <option value="roman_urdu">Roman Urdu</option>
            </select>

            {/* =================================================
                SPEAKER INFORMATION
            ================================================= */}

            <div className="speaker-information">
              <h3>Speaker Information (Required)</h3>

              <p>
                Complete all three fields before starting video
                analysis. If a speaker or detail is not listed,
                select Other and enter it manually.
              </p>

              {/* SPEAKER NAME */}

              <label htmlFor="speakerName">
                Speaker Name
              </label>

              <select
                id="speakerName"
                value={speakerName}
                required
                onChange={(event) => {
                  const value = event.target.value;

                  setSpeakerName(value);

                  if (value !== "Other") {
                    setSpeakerNameOther("");
                  }
                }}
              >
                <option value="">
                  Select a speaker (required)
                </option>

                {SPEAKER_NAME_OPTIONS.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}

                <option value="Other">Other</option>
              </select>

              {speakerName === "Other" && (
                <div className="speaker-other-field">
                  <label htmlFor="speakerNameOther">
                    Enter speaker name
                  </label>

                  <input
                    id="speakerNameOther"
                    type="text"
                    placeholder="Enter the speaker's full name"
                    value={speakerNameOther}
                    onChange={(event) =>
                      setSpeakerNameOther(event.target.value)
                    }
                    maxLength={255}
                    required
                  />
                </div>
              )}

              {/* SPEAKER PROFESSION */}

              <label htmlFor="speakerProfession">
                Speaker Profession
              </label>

              <select
                id="speakerProfession"
                value={speakerProfession}
                required
                onChange={(event) => {
                  const value = event.target.value;

                  setSpeakerProfession(value);

                  if (value !== "Other") {
                    setSpeakerProfessionOther("");
                  }
                }}
              >
                <option value="">
                  Select a profession (required)
                </option>

                {SPEAKER_PROFESSION_OPTIONS.map((profession) => (
                  <option key={profession} value={profession}>
                    {profession}
                  </option>
                ))}

                <option value="Other">Other</option>
              </select>

              {speakerProfession === "Other" && (
                <div className="speaker-other-field">
                  <label htmlFor="speakerProfessionOther">
                    Enter speaker profession
                  </label>

                  <input
                    id="speakerProfessionOther"
                    type="text"
                    placeholder="Enter the speaker's profession"
                    value={speakerProfessionOther}
                    onChange={(event) =>
                      setSpeakerProfessionOther(event.target.value)
                    }
                    maxLength={255}
                    required
                  />
                </div>
              )}

              {/* SPEAKER QUALIFICATION */}

              <label htmlFor="speakerQualification">
                Speaker Qualification
              </label>

              <select
                id="speakerQualification"
                value={speakerQualification}
                required
                onChange={(event) => {
                  const value = event.target.value;

                  setSpeakerQualification(value);

                  if (value !== "Other") {
                    setSpeakerQualificationOther("");
                  }
                }}
              >
                <option value="">
                  Select a qualification (required)
                </option>

                {SPEAKER_QUALIFICATION_OPTIONS.map(
                  (qualification) => (
                    <option
                      key={qualification}
                      value={qualification}
                    >
                      {qualification}
                    </option>
                  )
                )}

                <option value="Other">Other</option>
              </select>

              {speakerQualification === "Other" && (
                <div className="speaker-other-field">
                  <label htmlFor="speakerQualificationOther">
                    Enter speaker qualification
                  </label>

                  <input
                    id="speakerQualificationOther"
                    type="text"
                    placeholder="Enter the exact qualification(s)"
                    value={speakerQualificationOther}
                    onChange={(event) =>
                      setSpeakerQualificationOther(
                        event.target.value
                      )
                    }
                    maxLength={500}
                    required
                  />
                </div>
              )}

              {/* =================================================
                  CREATOR TYPE
                  Independent of Speaker Qualification
              ================================================= */}

              <label htmlFor="creatorType">
                Creator Type
              </label>

              <select
                id="creatorType"
                value={creatorType}
                required
                onChange={(event) => {
                  const value = event.target.value;

                  setCreatorType(value);

                  if (value !== "Other") {
                    setCreatorTypeOther("");
                  }
                }}
              >
                <option value="">
                  Select creator type (required)
                </option>

                <option value="Human">Human</option>
                <option value="AI">AI</option>
                <option value="Other">Other</option>
              </select>

              {creatorType === "Other" && (
                <div className="speaker-other-field">
                  <label htmlFor="creatorTypeOther">
                    Specify Creator Type
                  </label>

                  <input
                    id="creatorTypeOther"
                    type="text"
                    placeholder="e.g. Audio-only, animated character"
                    value={creatorTypeOther}
                    onChange={(event) =>
                      setCreatorTypeOther(event.target.value)
                    }
                    maxLength={255}
                    required
                  />
                </div>
              )}
            </div>

            {/* END SPEAKER INFORMATION */}

            <button type="submit" disabled={loading}>
              {loading
                ? "Processing video..."
                : "Start video analysis"}

              {!loading && <span> →</span>}
            </button>
          </form>

          <div className="supported-info">
            <span>✓</span> Supported video platforms

            <span className="platforms">
              YouTube · TikTok · Instagram · Vimeo
            </span>
          </div>
        </section>

        {/* ====================================================
            ERROR MESSAGE
        ==================================================== */}

        {error && (
          <section className="upload-card error-card">
            <h2>Analysis failed</h2>
            <p>{error}</p>
          </section>
        )}

        {/* ====================================================
            VIDEO AND TRANSCRIPT WORKSPACE
        ==================================================== */}

        {previewPlaying && (
          <section className="analysis-workspace">
            <div className="workspace-heading">
              <div>
                <span className="eyebrow">
                  LIVE ANALYSIS WORKSPACE
                </span>

                <h2>Video & Transcript</h2>

                <p>
                  Watch the video and follow its timestamped
                  transcript together.
                </p>
              </div>

              {result && (
                <span className="saved-badge">✓ Saved</span>
              )}
            </div>

            <div className="workspace-grid">
              {/* VIDEO PLAYER */}

              <div className="player-panel">
                <div className="panel-heading">
                  <h3>Video Player</h3>

                  <span className="player-time">
                    {formatTime(playbackTime)}
                  </span>
                </div>

                {videoDetails?.embedUrl ? (
                  <div className="video-frame">
                    <iframe
                      ref={iframeRef}
                      key={videoDetails.embedUrl}
                      src={videoDetails.embedUrl}
                      title="ClaimCheck Video Player"
                      allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
                      allowFullScreen
                    />
                  </div>
                ) : (
                  <div className="player-unavailable">
                    <p>Preview is not available for this URL.</p>

                    <a
                      href={videoUrl}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open original video
                    </a>
                  </div>
                )}

                <div className="player-status">
                  <span className="status-dot"></span>

                  {videoDetails?.provider === "youtube"
                    ? playerStatus
                    : "Video preview loaded · Playback-time sync depends on platform"}
                </div>

                {videoDetails?.provider !== "youtube" && (
                  <p className="sync-note">
                    Automatic timestamp synchronization and
                    click-to-seek are currently supported for
                    YouTube. Other platform embeds may restrict
                    playback-time access.
                  </p>
                )}
              </div>

              {/* TIMESTAMPED TRANSCRIPT */}

              <div className="transcript-panel">
                <div className="panel-heading">
                  <h3>Timestamped Transcript</h3>

                  <span className="segment-count">
                    {result
                      ? `${segments.length} segments`
                      : "Preparing..."}
                  </span>
                </div>

                {loading && !result ? (
                  <div className="transcript-empty">
                    <div className="loading-indicator"></div>

                    <h3>Generating transcript</h3>

                    <p>
                      Your video is playing. Transcript segments
                      will appear here when analysis is complete.
                    </p>
                  </div>
                ) : segments.length > 0 ? (
                  <div className="transcript-list">
                    {segments.map((segment, index) => (
                      <button
                        type="button"
                        key={`${segment.start}-${segment.end}-${index}`}
                        ref={(element) => {
                          segmentRefs.current[index] = element;
                        }}
                        className={`transcript-segment ${
                          index === activeSegmentIndex
                            ? "active"
                            : ""
                        }`}
                        onClick={() =>
                          handleSegmentClick(segment)
                        }
                        title={
                          videoDetails?.provider === "youtube"
                            ? "Jump video to this segment"
                            : "Timestamp seeking is not available for this platform"
                        }
                      >
                        <span className="transcript-segment-meta">
                          <span>
                            {formatTime(segment.start)} –{" "}
                            {formatTime(segment.end)}
                          </span>
                        </span>

                        <span className="transcript-segment-text">
                          {segment.text}
                        </span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="transcript-empty">
                    <h3>Transcript will appear here</h3>

                    <p>
                      Start the analysis to generate timestamped
                      transcript segments alongside the video.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {activeSegment && (
              <div className="active-segment-footer">
                <strong>Currently active segment:</strong>{" "}
                {formatTime(activeSegment.start)} –{" "}
                {formatTime(activeSegment.end)}
              </div>
            )}
          </section>
        )}

        {/* ====================================================
            ANALYSIS RESULTS AND HUMAN REVIEW
        ==================================================== */}

        {result && (
          <>
            {/* =================================================
                HUMAN CLAIM REVIEW
            ================================================= */}

            {allClaims.length > 0 && (
              <section className="review-card">
                <div className="review-header">
                  <div className="review-heading">
                    <h2>Human Claim Review</h2>

                    <p>
                      Research annotation · Independent reviewer
                      assessment
                    </p>
                  </div>

                  <div className="review-icon">✓</div>
                </div>

                {/* SUCCESS STATE */}

                {reviewStatus ? (
                  <div className="review-success">
                    <div className="review-success-icon">✓</div>

                    <h3>Review Successfully Completed</h3>

                    <p>{reviewStatus}</p>
                  </div>
                ) : (
                  <>
                    {/* REVIEW PROGRESS */}

                    <div className="review-progress">
                      <div className="review-progress-info">
                        <span>Review progress</span>

                        <strong>
                          Claim {reviewIndex + 1} of{" "}
                          {allClaims.length}
                        </strong>
                      </div>

                      <div className="review-progress-track">
                        <div
                          className="review-progress-fill"
                          style={{
                            width: `${
                              ((reviewIndex + 1) /
                                allClaims.length) *
                              100
                            }%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* CURRENT CLAIM */}

                    <div className="review-claim-box">
                      <span className="review-claim-label">
                        Extracted factual claim
                      </span>

                      <h3>
                        {currentClaim?.text ||
                          currentClaim?.claim_text}
                      </h3>

                      <div className="review-timestamp">
                        Video timestamp:{" "}
                        {currentClaim?.segmentStart}s –{" "}
                        {currentClaim?.segmentEnd}s
                      </div>
                    </div>

                    {/* PROPOSED CLEAR MEANING */}

                    {currentClaim?.proposed_clear_meaning && (
                      <div className="review-claim-box">
                        <span className="review-claim-label">
                          Proposed Clear Meaning
                        </span>

                        <p>
                          {currentClaim.proposed_clear_meaning}
                        </p>
                      </div>
                    )}

                    {/* REVIEW DECISION */}

                    <h4 className="review-options-title">
                      Select your review decision
                    </h4>

                    <div className="review-options">
                      {reviewOptions.map((option) => (
                        <label
                          key={option.value}
                          className={`review-option ${
                            currentReview.reviewed_label ===
                            option.value
                              ? "selected"
                              : ""
                          }`}
                        >
                          <input
                            type="radio"
                            name={`review-${currentClaim?.claim_id}`}
                            value={option.value}
                            checked={
                              currentReview.reviewed_label ===
                              option.value
                            }
                            onChange={() =>
                              handleDecision(option.value)
                            }
                          />

                          <span className="review-option-content">
                            <strong>{option.label}</strong>

                            <small>{option.description}</small>
                          </span>
                        </label>
                      ))}
                    </div>


{/* =================================================
    EVIDENCE VERIFICATION
    Only displayed for Agree
================================================= */}

{currentReview.reviewed_label === "agree" && (
  <div className="evidence-review-section">
    <h4 className="review-options-title">
      Q2. What is the truth status of this claim?
    </h4>

    {/* TRUTH STATUS */}

    <div className="review-options">
      {truthStatusOptions.map((option) => (
        <label
          key={option.value}
          className={`review-option ${
            currentReview.truth_status === option.value
              ? "selected"
              : ""
          }`}
        >
          <input
            type="radio"
            name={`truth-${currentClaim.claim_id}`}
            value={option.value}
            checked={
              currentReview.truth_status === option.value
            }
            onChange={() =>
              handleReviewFieldChange(
                "truth_status",
                option.value
              )
            }
          />

          <span className="review-option-content">
            <strong>{option.label}</strong>
          </span>
        </label>
      ))}
    </div>

    {/* SOURCE VERIFICATION */}

    {currentReview.truth_status && (
      <div className="source-verification-section">
        <h4 className="review-options-title">
          Q3. Source Verification
        </h4>

        <p>
          Provide a reliable source and explain how its evidence
          supports your selected truth status. If verification
          is not possible, select Cannot be verified.
        </p>

        {/* SOURCE CATEGORY */}

        <label htmlFor="sourceCategory">
          Source Category
        </label>

        <select
          id="sourceCategory"
          value={currentReview.source_category || ""}
          onChange={(event) =>
            handleReviewFieldChange(
              "source_category",
              event.target.value
            )
          }
          required
        >
          <option value="">
            Select source category
          </option>

          <option value="Pakistan Dietary Guidelines for Better Nutrition (2018)">
            Pakistan Dietary Guidelines for Better Nutrition (2018)
          </option>

          <option value="WHO Guidelines">
            WHO Guidelines
          </option>

          <option value="Disease Guidelines">
            Disease Guidelines
          </option>

          <option value="Supplement and Health Claims References">
            Supplement and Health Claims References
          </option>

          <option value="Systematic Reviews and Meta-Analyses">
            Systematic Reviews and Meta-Analyses
          </option>

          <option value="Other">
            Other
          </option>

          <option value="Cannot be verified">
            Cannot be verified
          </option>
        </select>

        {/* SOURCE DETAILS
            Hidden when Cannot be verified is selected
        */}

        {currentReview.source_category !== "Cannot be verified" &&
          currentReview.source_category && (
            <>
              <label htmlFor="sourceName">
                Actual Source Name / Document
              </label>

              <input
                id="sourceName"
                type="text"
                placeholder="e.g. WHO Guidelines on Sodium Intake (2012)"
                value={currentReview.source_name || ""}
                onChange={(event) =>
                  handleReviewFieldChange(
                    "source_name",
                    event.target.value
                  )
                }
                required
              />

              <label htmlFor="sourceUrl">
                Source URL
              </label>

              <input
                id="sourceUrl"
                type="url"
                placeholder="https://..."
                value={currentReview.source_url || ""}
                onChange={(event) =>
                  handleReviewFieldChange(
                    "source_url",
                    event.target.value
                  )
                }
                required
              />

              <label htmlFor="evidenceText">
                Evidence Explanation
              </label>

              <textarea
                id="evidenceText"
                placeholder="Explain what the source says and how it supports your truth status."
                value={currentReview.evidence_text || ""}
                onChange={(event) =>
                  handleReviewFieldChange(
                    "evidence_text",
                    event.target.value
                  )
                }
                rows={4}
                required
              />
            </>
          )}
      </div>
    )}
  </div>
)}


                    {/* REVIEW ERRORS */}

                    {reviewError && (
                      <div className="review-error">
                        {reviewError}
                      </div>
                    )}

                    {/* =================================================
                        REVIEW NAVIGATION
                    ================================================= */}

                    <div className="review-navigation">
                      <button
                        type="button"
                        className="review-next review-previous"
                        onClick={handlePreviousClaim}
                        disabled={
                          reviewIndex === 0 || reviewSaving
                        }
                      >
                        ← Previous Claim
                      </button>

                      {reviewIndex < allClaims.length - 1 ? (
                        <button
                          type="button"
                          className="review-next"
                          onClick={handleNextClaim}
                          disabled={
                            reviewSaving ||
                            !currentReview.reviewed_label
                          }
                        >
                          Next Claim →
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="review-next"
                          onClick={handleReviewSubmit}
                          disabled={
                            reviewSaving ||
                            !currentReview.reviewed_label
                          }
                        >
                          {reviewSaving
                            ? "Saving Reviews..."
                            : "✓ Done — Submit Reviews"}
                        </button>
                      )}
                    </div>
                  </>
                )}
              </section>
            )}

            {/* NO CLAIMS */}

            {allClaims.length === 0 && (
              <section className="upload-card">
                <h2>Human Review</h2>

                <p>
                  No database-backed claims are available for
                  human review in this video.
                </p>
              </section>
            )}

            {/* =================================================
                ANALYSIS RESULTS
            ================================================= */}

            <section className="upload-card results-card">
              <div className="card-heading">
                <div>
                  <h2>Analysis Results</h2>

                  <p>
                    Video successfully processed and saved.
                  </p>
                </div>

                <div className="video-icon">✓</div>
              </div>

              <h3>{result.video_title}</h3>

              <p>
                <strong>Language:</strong> {result.language}
              </p>

              <h3>Full Transcript</h3>

              <p className="full-transcript">{result.text}</p>

              <h3>
                Extracted Segments (
                {result.analysis?.total_segments || 0})
              </h3>

              {segments.map((segment, index) => (
                <div
                  className="workflow-card result-segment"
                  key={index}
                >
                  <h3>
                    Segment {index + 1}: {segment.start}s –{" "}
                    {segment.end}s
                  </h3>

                  <p>
                    <strong>Category:</strong> {segment.category}
                  </p>

                  <p>{segment.text}</p>

                  <h4>Claims</h4>

                  {segment.claims?.length > 0 ? (
                    segment.claims.map((claim, claimIndex) => (
                      <div
                        key={claim.claim_id || claimIndex}
                        className="result-claim"
                      >
                        <p>
                          {claim.text || claim.claim_text}
                        </p>

                        {claim.search_queries?.length > 0 && (
                          <ul>
                            {claim.search_queries.map(
                              (query, queryIndex) => (
                                <li key={queryIndex}>
                                  {query}
                                </li>
                              )
                            )}
                          </ul>
                        )}
                      </div>
                    ))
                  ) : (
                    <p>
                      No factual claims extracted from this segment.
                    </p>
                  )}
                </div>
              ))}
            </section>
          </>
        )}

        {/* ====================================================
            HOW IT WORKS
        ==================================================== */}

        <section className="workflow-section">
          <h2>How it works</h2>

          <div className="workflow-grid">
            <div className="workflow-card">
              <span className="step-number">01</span>
              <h3>Transcription</h3>
              <p>
                Generate timestamped transcripts from video audio.
              </p>
            </div>

            <div className="workflow-card">
              <span className="step-number">02</span>
              <h3>Claim extraction</h3>
              <p>
                Identify factual claims using AI-powered analysis.
              </p>
            </div>

            <div className="workflow-card">
              <span className="step-number">03</span>
              <h3>Human review</h3>
              <p>
                Review claims and assign structured annotations.
              </p>
            </div>

            <div className="workflow-card">
              <span className="step-number">04</span>
              <h3>Research dataset</h3>
              <p>
                Save reviewed data for future machine learning.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer>
        ClaimCheck Video AI · Evidence-based video research platform
      </footer>
    </div>
  );
}

export default App;
