
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function checkBackendHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("Backend connection failed");
  }

  return await response.json();
}


export async function processVideo(url, mode = "auto", speakerMetadata = {}) {
  const response = await fetch(
    `${API_BASE_URL}/analyze/video-to-claims`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url,
        mode,

        // Speaker information
        speaker_name: speakerMetadata.speaker_name || null,
        speaker_profession: speakerMetadata.speaker_profession || null,
        speaker_qualification: speakerMetadata.speaker_qualification || null,

        // Creator Type information
        creator_type: speakerMetadata.creator_type || null,
        creator_type_other: speakerMetadata.creator_type_other || null,
      }),
    }
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = data.detail;

    let message =
      typeof detail === "string"
        ? detail
        : detail?.message ||
          detail?.detail ||
          "Video processing failed";

    if (
      response.status === 409 &&
      !detail?.message &&
      typeof detail !== "string"
    ) {
      message =
        "This video has already been submitted. Please submit a different video.";
    }

    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;

    throw error;
  }

  return data;
}

// Submit human reviews to PostgreSQL through the backend.


export async function submitClaimReviews(video_id, reviews) {
  const response = await fetch(
    `${API_BASE_URL}/reviews/submit`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_id,
        reviews,
      }),
    }
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = data.detail;

    let message = "Failed to submit claim reviews";

    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((error) => {
          const field = error.loc?.join(" → ") || "Unknown field";
          return `${field}: ${error.msg}`;
        })
        .join("\n");
    } else if (detail?.message) {
      message = detail.message;
    }

    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;

    throw error;
  }

  return data;
}

