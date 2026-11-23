(() => {
  "use strict";

  // Full-scene metrics from the sensor and multiview rows of the ScanNet++ v2 report.
  const scenes = {
    c0f5742640: {
      name: "Apartment", frames: 468, frame: "003100", rgb: "the apartment's corner windows",
      fscore: [40.90, 68.21], mae: [8.53, 8.89]
    },
    "791a5c253d": {
      name: "Conference room", frames: 555, frame: "002230", rgb: "the conference room windows and blinds",
      fscore: [47.52, 66.38], mae: [17.72, 11.65]
    },
    "8a20d62ac0": {
      name: "Office", frames: 295, frame: "002450", rgb: "the windows behind the office desk",
      fscore: [77.63, 89.25], mae: [5.21, 6.24]
    }
  };
  const byId = (id) => document.getElementById(id);
  const controls = byId("recon-controls");
  const display = byId("recon-display");
  const status = byId("recon-status");
  const slider = byId("recon-split");
  let shown = { scene: "c0f5742640", view: "full" };
  let request = 0;

  async function prepareImage(id, src, alt) {
    const previous = byId(id);
    if (previous.getAttribute("src") === src && previous.naturalWidth) return previous;
    const image = new Image(Number(previous.getAttribute("width")), Number(previous.getAttribute("height")));
    image.id = id;
    image.alt = alt;
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = src;
    });
    if (image.decode) await image.decode();
    return image;
  }

  function restoreSelection() {
    for (const key of ["scene", "view"]) {
      controls.querySelector(`input[name="recon-${key}"][value="${shown[key]}"]`).checked = true;
    }
  }

  async function changeSelection() {
    const serial = ++request;
    const scene = controls.querySelector('input[name="recon-scene"]:checked').value;
    const view = controls.querySelector('input[name="recon-view"]:checked').value;
    if (shown.scene === scene && shown.view === view) {
      display.removeAttribute("aria-busy");
      status.textContent = "";
      return;
    }
    const data = scenes[scene];
    const base = `pics/reconstructions/${scene}/`;
    const title = data.name + (view === "full" ? " full scene" : " glass detail");
    display.setAttribute("aria-busy", "true");
    status.textContent = `Loading ${title.toLowerCase()}…`;

    try {
      // Commit both renders, the reference, and the readout together, even on rapid scene changes.
      const images = await Promise.all([
        prepareImage("recon-raw", base + view + "-raw.webp", title + " reconstructed from raw sensor depth"),
        prepareImage("recon-ours", base + view + "-ours.webp", title + " reconstructed by GlassRecon"),
        prepareImage("recon-rgb", base + "rgb.webp", "RGB input looking toward " + data.rgb)
      ]);
      if (serial !== request) return;
      for (const image of images) {
        const previous = byId(image.id);
        if (previous !== image) previous.replaceWith(image);
      }
      byId("recon-name").textContent = data.name;
      byId("recon-id").textContent = scene;
      byId("recon-frames").textContent = data.frames;
      byId("recon-frame").textContent = data.frame;
      byId("recon-raw-f").textContent = data.fscore[0].toFixed(2) + "%";
      byId("recon-ours-f").textContent = data.fscore[1].toFixed(2) + "%";
      byId("recon-raw-mae").textContent = data.mae[0].toFixed(2) + " cm";
      byId("recon-ours-mae").textContent = data.mae[1].toFixed(2) + " cm";
      shown = { scene, view };
      status.textContent = "";
    } catch {
      if (serial !== request) return;
      restoreSelection();
      status.textContent = "Image loading failed. Select the scene or view again to retry.";
    }
    display.removeAttribute("aria-busy");
  }

  function updateDivider() {
    const percent = slider.valueAsNumber;
    byId("recon-stage").style.setProperty("--split", percent + "%");
    slider.setAttribute("aria-valuetext", `${percent}% raw sensor, ${100 - percent}% GlassRecon`);
  }

  controls.addEventListener("change", changeSelection);
  slider.addEventListener("input", updateDivider);
  controls.hidden = false;
  slider.disabled = false;
  restoreSelection();
  updateDivider();

  const video = document.querySelector(".recon-extras video");
  const videoPanel = video.closest("details");
  videoPanel.addEventListener("toggle", () => {
    if (!videoPanel.open) video.pause();
  });
})();
