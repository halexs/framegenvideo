/* StreamerFrames UI. All server text goes through textContent; nothing is inserted as HTML. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  async function api(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    return data;
  }

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c) node.append(c);
    return node;
  }

  function fmtTime(sec) {
    if (sec == null || !isFinite(sec)) return "?";
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
  }

  function fmtDur(sec) {
    if (sec == null || !isFinite(sec)) return "?";
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
    return h ? `${h}h${String(m).padStart(2, "0")}m` : `${Math.max(1, m)}m`;
  }

  function fps(frac) {
    if (!frac) return "";
    const [n, d] = String(frac).split("/").map(Number);
    return (d ? n / d : n).toFixed(3).replace(/\.?0+$/, "");
  }

  const query = new URLSearchParams(location.search);
  const pathId = () => location.pathname.split("/").filter(Boolean)[1];

  /* ---------- Library ---------- */

  function jobBadge(job) {
    if (!job) return null;
    const p = job.progress || {};
    if (job.state === "running") {
      const ratio = p.realtime_ratio != null ? ` · ${p.realtime_ratio.toFixed(2)}×` : "";
      const eta = p.eta_seconds != null ? ` · ETA ${fmtDur(p.eta_seconds)}` : "";
      return el("span", { class: "badge run", text: `Generating ${job.percent ?? 0}%${ratio}${eta} (${job.kind})` });
    }
    return el("span", { class: "badge", text: `Queued (${job.kind})` });
  }

  function cacheBadge(c) {
    const name = c.profile_name || c.profile_id;
    if (c.status === "complete") {
      const exp = c.export_state === "running" ? " · exporting" : c.has_export ? " · exported" : "";
      return el("span", { class: "badge ok", text: `${name}: ready ${fps(c.out_fps)} fps${exp}` });
    }
    if (c.status === "failed") return el("span", { class: "badge err", text: `${name}: failed`, title: c.error || "" });
    if (c.status === "running") return null; // the job badge says it
    return el("span", { class: "badge warn", text: `${name}: paused ${c.percent}%` });
  }

  function libraryPage() {
    let videos = [];
    const filter = $("filter");

    function render() {
      const q = filter.value.trim().toLowerCase();
      const rows = $("rows");
      rows.replaceChildren();
      const shown = videos.filter((v) => !q || v.title.toLowerCase().includes(q));
      if (!shown.length) {
        rows.append(el("tr", {}, el("td", { colspan: "4", class: "muted", text: videos.length ? "No match." : "No movies found. Check movies_roots in streamerframes.toml." })));
        return;
      }
      for (const v of shown) {
        const src = v.width ? `${v.width}×${v.height} · ${fps(v.fps)} fps · ${fmtDur(v.duration)}` : v.container;
        const badges = el("td");
        const jb = jobBadge(v.job);
        if (jb) badges.append(jb);
        for (const c of v.caches) { const b = cacheBadge(c); if (b) badges.append(b); }
        if (!badges.childNodes.length) badges.append(el("span", { class: "muted", text: "—" }));

        const actions = el("td");
        actions.append(el("a", { class: "btn", href: `/watch/${v.video_id}?mode=original`, text: "Original" }));
        actions.append(el("a", { class: "btn", href: `/watch/${v.video_id}?mode=framegen&profile=realtime`, text: "Framegen" }));
        actions.append(el("button", {
          text: "Generate in background", title: "Queue an offline job with the quality profile",
          onclick: () => api("POST", "/api/jobs", { video_id: v.video_id, kind: "offline", profile: "quality" }).then(load, alert),
        }));
        for (const c of v.caches) {
          if (c.status === "complete" && !c.has_export && c.export_state !== "running")
            actions.append(el("button", { text: `Export ${c.profile_name || ""}`,
              onclick: () => api("POST", `/api/videos/${v.video_id}/export`, { profile_id: c.profile_id }).then(load, alert) }));
          if (c.has_export)
            actions.append(el("a", { class: "btn", href: `/exports/${v.video_id}/${c.profile_id}`, text: `Download ${c.profile_name || ""}` }));
        }
        if (v.job) actions.append(el("button", { text: "Cancel job", onclick: () => api("DELETE", `/api/jobs/${v.job.id}`).then(load, alert) }));
        else if (v.caches.some((c) => c.status === "paused" || c.status === "failed"))
          actions.append(el("button", { text: "Resume", onclick: () => {
            const c = v.caches.find((x) => x.status === "paused" || x.status === "failed");
            api("POST", "/api/jobs", { video_id: v.video_id, kind: "offline", profile: c.profile_name.split("+")[0] }).then(load, alert);
          } }));
        if (v.caches.length && !v.job)
          actions.append(el("button", { text: "Delete cache", onclick: () => {
            if (confirm(`Delete generated frames for "${v.title}"?`)) api("DELETE", `/api/cache/${v.video_id}`).then(load, alert);
          } }));
        rows.append(el("tr", {}, el("td", { text: v.title }), el("td", { class: "muted hide-narrow", text: src }), badges, actions));
      }
    }

    async function load(refresh) {
      try {
        const data = await api("GET", "/api/library" + (refresh === true ? "?refresh=1" : ""));
        videos = data.videos;
        const r = data.running;
        $("gpu").textContent = r ? `GPU: ${r.kind} job ${r.percent ?? 0}%` : "GPU idle";
        render();
      } catch (e) {
        $("rows").replaceChildren(el("tr", {}, el("td", { colspan: "4", class: "err", text: `Error: ${e.message}` })));
      }
    }

    filter.addEventListener("input", render);
    $("refresh").addEventListener("click", () => load(true));
    load();
    setInterval(load, 3000);
  }

  /* ---------- Playback helpers ---------- */

  function attachHls(video, url, opts) {
    if (window.Hls && Hls.isSupported()) {
      const hls = new Hls(Object.assign({
        startPosition: 0,            // never jump to the "live" edge of a growing playlist
        maxBufferLength: 60,
        fragLoadingMaxRetry: 30,     // segments may be long-polled or 503 while generating (Phase 4)
        fragLoadingRetryDelay: 1000,
        fragLoadingMaxRetryTimeout: 4000,
        manifestLoadingMaxRetry: 10,
        levelLoadingMaxRetry: 10,
      }, opts || {}));
      hls.loadSource(url);
      hls.attachMedia(video);
      return hls;
    }
    if (video.canPlayType("application/vnd.apple.mpegurl")) { video.src = url; return null; }
    throw new Error("This browser cannot play HLS.");
  }

  /** Generated video seconds per wall second, smoothed over the last 60 s of progress samples. */
  class RateMeter {
    constructor() { this.samples = []; }
    add(seconds) {
      const now = performance.now() / 1000;
      this.samples.push([now, seconds]);
      while (this.samples.length > 2 && now - this.samples[0][0] > 60) this.samples.shift();
    }
    rate(fallback) {
      if (this.samples.length < 2) return fallback;
      const [t0, s0] = this.samples[0], [t1, s1] = this.samples[this.samples.length - 1];
      return t1 - t0 >= 5 ? (s1 - s0) / (t1 - t0) : fallback;
    }
  }

  /**
   * Can playback from `pos` run to the end without stalling? Buffered lead B = frontier - pos,
   * remaining D = duration - pos, generation rate r (×real time, with 10% margin).
   * Frontier stays ahead iff B >= D·(1 − r). Waiting w seconds adds r·w to B.
   */
  function safeStart(frontier, duration, pos, r) {
    const B = Math.max(0, frontier - pos), D = Math.max(0, duration - pos);
    if (frontier >= duration - 0.01) return { ok: true, wait: 0 };
    const re = (r || 0) * 0.9;
    if (re <= 0) return { ok: false, wait: Infinity };
    const need = D * Math.max(0, 1 - re);
    if (B >= need && B >= 8) return { ok: true, wait: 0 };
    return { ok: false, wait: Math.max(need - B, 8 - B) / re };
  }

  /* ---------- Watch ---------- */

  function watchPage() {
    const vid = pathId();
    const mode = query.get("mode") || "framegen";
    const profile = query.get("profile") || "realtime";
    const video = $("video");
    $("to-original").href = `/watch/${vid}?mode=original`;
    $("to-framegen").href = `/watch/${vid}?mode=framegen&profile=${encodeURIComponent(profile)}`;
    $("to-compare").href = `/compare/${vid}?profile=${encodeURIComponent(profile)}`;
    api("GET", `/api/videos/${vid}`).then((v) => {
      $("title").textContent = v.title;
      document.title = `${v.title} · StreamerFrames`;
    }).catch(() => {});

    if (mode === "original") {
      video.src = `/media/original/${vid}`;
      video.addEventListener("error", () => {
        $("status").hidden = false;
        $("s-main").textContent = "This browser can't play the original file (MKV and some codecs aren't supported). Try Framegen, which streams H.264.";
      });
      video.play().catch(() => {});
      return;
    }

    $("status").hidden = false;
    $("s-main").textContent = "Starting generation…";
    let started = false, finished = false, job = null, frontier = 0, duration = null;
    const meter = new RateMeter();

    function start(pid) {
      if (started) return;
      started = true;
      $("play-now").hidden = true;
      try {
        attachHls(video, `/hls/${vid}/${pid}/master.m3u8`);
        video.play().catch(() => {});
      } catch (e) { $("s-advice").textContent = e.message; }
    }

    video.addEventListener("waiting", () => {
      if (!finished && job && job.state === "running") $("s-advice").textContent = `Generating… (ready up to ${fmtTime(frontier)})`;
    });
    $("play-now").addEventListener("click", () => job && start(job.profile_id));
    $("cancel").addEventListener("click", () => job && job.id && api("DELETE", `/api/jobs/${job.id}`).catch(alert));

    async function poll() {
      try {
        job = job && job.id ? await api("GET", `/api/jobs/${job.id}`) : await api("POST", "/api/jobs", { video_id: vid, kind: "stream", profile });
      } catch (e) {
        $("s-main").textContent = `Error: ${e.message}`;
        return setTimeout(poll, 5000);
      }
      if (job.state === "complete" || job.manifest_status === "complete") {
        finished = true;
        $("s-main").textContent = "Ready.";
        $("s-detail").textContent = ""; $("s-advice").textContent = ""; $("cancel").hidden = true;
        start(job.profile_id);
        return;
      }
      if (job.state === "failed") {
        $("s-main").textContent = `Generation failed: ${job.error || "see worker.log"}`;
        return;
      }
      if (job.state === "cancelled" || job.state === "paused") {
        $("s-main").textContent = `Generation ${job.state}.`;
        $("cancel").hidden = true;
        return;
      }
      frontier = job.contiguous_seconds || 0;
      duration = job.duration_seconds || duration;
      meter.add(frontier);
      const p = job.progress || {};
      const r = meter.rate(p.realtime_ratio || 0);
      $("cancel").hidden = false;
      if (job.state === "queued") {
        $("s-main").textContent = "Waiting for the GPU…";
      } else {
        $("s-main").textContent = `Generated ${fmtTime(frontier)} of ${fmtTime(duration)} · ${r.toFixed(2)}× real time` +
          (p.eta_seconds != null ? ` · done in ${fmtDur(p.eta_seconds)}` : "");
        $("s-detail").textContent = `${job.contiguous_segments || 0}/${job.segments_total || "?"} segments · source ${p.src_fps_measured ?? "?"} fps`;
      }
      if (duration) {
        const pos = started ? video.currentTime : 0;
        const s = safeStart(frontier, duration, pos, r);
        if (s.ok) $("s-advice").textContent = "Playing now will run without stalling.";
        else $("s-advice").textContent = isFinite(s.wait)
          ? `Wait ~${fmtDur(s.wait)} for stall-free playback (or play now).`
          : "Measuring generation speed…";
        const ready = (job.contiguous_segments || 0) >= 3;
        if (!started && ready && s.ok) start(job.profile_id);
        $("play-now").hidden = started || !ready;
      }
      setTimeout(poll, 2000);
    }
    poll();
  }

  /* ---------- Compare ---------- */

  function comparePage() {
    const vid = pathId();
    const profile = query.get("profile") || "realtime";
    const left = $("left"), right = $("right");
    api("GET", `/api/videos/${vid}`).then((v) => { $("title").textContent = v.title; }).catch(() => {});
    left.src = `/media/original/${vid}`;

    let job = null;
    async function poll() {
      job = job && job.id ? await api("GET", `/api/jobs/${job.id}`) : await api("POST", "/api/jobs", { video_id: vid, kind: "stream", profile });
      if ((job.contiguous_segments || 0) >= 3 || job.state === "complete") {
        attachHls(right, `/hls/${vid}/${job.profile_id}/master.m3u8`);
        return;
      }
      $("status").hidden = false;
      $("s-main").textContent = `Generating… ${fmtTime(job.contiguous_seconds || 0)} ready`;
      setTimeout(poll, 2000);
    }
    poll().catch((e) => { $("status").hidden = false; $("s-main").textContent = `Error: ${e.message}`; });

    right.addEventListener("play", () => left.play().catch(() => {}));
    right.addEventListener("pause", () => left.pause());
    right.addEventListener("seeked", () => { left.currentTime = right.currentTime; });
    setInterval(() => {
      if (!right.paused && Math.abs(left.currentTime - right.currentTime) > 0.15) left.currentTime = right.currentTime;
    }, 1000);
    $("toggle").addEventListener("click", () => (right.paused ? right.play() : right.pause()));
  }

  window.SF = { libraryPage, watchPage, comparePage, safeStart, RateMeter, api };
})();
