// Camera lifecycle: getUserMedia preview (CSS-mirrored) + unmirrored JPEG frame capture.

export class CameraError extends Error {
  constructor(kind, message) {
    super(message);
    this.kind = kind; // 'unsupported' | 'denied' | 'notfound' | 'busy'
  }
}

export class Camera {
  constructor(videoEl) {
    this.video = videoEl;
    this.stream = null;
    window.addEventListener('pagehide', () => this.stop());
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new CameraError(
        'unsupported',
        'Camera API unavailable. Open the app via http://localhost (not file:// or a LAN IP).'
      );
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
    } catch (err) {
      const name = err && err.name;
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        throw new CameraError(
          'denied',
          'Camera permission was denied. Re-enable it in your browser’s site settings, then try again.'
        );
      }
      if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        throw new CameraError('notfound', 'No camera was found on this device.');
      }
      if (name === 'NotReadableError' || name === 'AbortError') {
        throw new CameraError(
          'busy',
          'The camera is in use by another app. Close it and try again.'
        );
      }
      throw new CameraError('busy', `Could not start the camera (${name || err}).`);
    }
    this.video.srcObject = this.stream;
    await new Promise((resolve) => {
      if (this.video.readyState >= 1 && this.video.videoWidth > 0) return resolve();
      this.video.addEventListener('loadedmetadata', resolve, { once: true });
    });
    await this.video.play().catch(() => {});
  }

  get ready() {
    return Boolean(this.stream) && this.video.videoWidth > 0;
  }

  /** Capture `count` unmirrored JPEG Blobs off a hidden canvas. */
  async captureFrames(count = 6, intervalMs = 200) {
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const blobs = [];
    for (let i = 0; i < count; i += 1) {
      ctx.drawImage(this.video, 0, 0, w, h); // ignores the CSS mirror — frames are unmirrored
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
      if (blob) blobs.push(blob);
      if (i < count - 1) await new Promise((r) => setTimeout(r, intervalMs));
    }
    return { blobs, width: w, height: h, frameIntervalMs: intervalMs };
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      this.video.srcObject = null;
    }
  }
}
