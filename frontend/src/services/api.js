const API_BASE_URL = "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep fallback detail string.
    }
    throw new Error(detail);
  }

  return response.json();
}

export async function fetchSignals(payload) {
  return request("/api/signals", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchTopSignals(topN = 3, maxScan = 3) {
  return request(`/api/signals/top?top_n=${topN}&max_scan=${maxScan}`);
}

export async function fetchChart(ticker, useVlm = true) {
  return request(`/api/charts/${encodeURIComponent(ticker)}?use_vlm=${useVlm}`);
}

export async function fetchRadar(topN = 10) {
  return request(`/api/radar?top_n=${topN}`);
}

export async function queryIntelligence(payload) {
  return request("/api/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function queryVoice(file, ticker = "") {
  const formData = new FormData();
  formData.append("file", file);

  const querySuffix = ticker ? `?ticker=${encodeURIComponent(ticker)}` : "";
  const response = await fetch(`${API_BASE_URL}/api/voice${querySuffix}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail = `Voice request failed: ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep fallback detail string.
    }
    throw new Error(detail);
  }

  return response.json();
}

export async function fetchLatestAlerts(limit = 10) {
  return request(`/api/alerts/latest?limit=${limit}`);
}

export async function fetchHealth() {
  return request("/api/health");
}
