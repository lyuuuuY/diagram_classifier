const dropZone = document.querySelector("#dropZone");
const fileInput = document.querySelector("#fileInput");
const emptyState = document.querySelector("#emptyState");
const previewState = document.querySelector("#previewState");
const previewImage = document.querySelector("#previewImage");
const fileName = document.querySelector("#fileName");
const fileDetails = document.querySelector("#fileDetails");
const message = document.querySelector("#message");
const clearButton = document.querySelector("#clearButton");
const classifyButton = document.querySelector("#classifyButton");
const resultCard = document.querySelector("#resultCard");
const resultLabel = document.querySelector("#resultLabel");
const resultSymbol = document.querySelector("#resultSymbol");
const serverState = document.querySelector("#serverState");

const MAX_FILE_SIZE = 5 * 1024 * 1024;
const LABEL_DETAILS = {
  equation: { label: "Equation", symbol: "∫ dx" },
  graph: { label: "Connected graph", symbol: "⌬" },
  lewis: { label: "Lewis structure", symbol: "H—O" },
};

let selectedFile = null;
let previewUrl = null;

async function checkServer() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Server unavailable");
    serverState.className = "server-state ready";
    serverState.lastElementChild.textContent = "Local model ready";
  } catch {
    serverState.className = "server-state error";
    serverState.lastElementChild.textContent = "Server unavailable";
  }
}

function showMessage(text = "") {
  message.textContent = text;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

async function getImageSize(file) {
  if ("createImageBitmap" in window) {
    const bitmap = await createImageBitmap(file);
    const size = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return size;
  }

  return new Promise((resolve, reject) => {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);
    image.onload = () => {
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
      URL.revokeObjectURL(objectUrl);
    };
    image.onerror = () => {
      reject(new Error("The selected file is not a valid image."));
      URL.revokeObjectURL(objectUrl);
    };
    image.src = objectUrl;
  });
}

async function selectFile(file) {
  clearSelection();
  if (!file) return;

  const isPng =
    file.type === "image/png" || file.name.toLowerCase().endsWith(".png");
  if (!isPng) {
    showMessage("Please select a PNG image.");
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showMessage("The PNG must be smaller than 5 MB.");
    return;
  }

  try {
    const size = await getImageSize(file);
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    fileName.textContent = file.name;
    fileDetails.textContent =
      `${size.width}×${size.height} · ${formatBytes(file.size)}`;
    emptyState.hidden = true;
    previewState.hidden = false;
    clearButton.disabled = false;
    classifyButton.disabled = false;
    showMessage();
  } catch (error) {
    showMessage(error.message || "The selected image could not be read.");
  }
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewImage.removeAttribute("src");
  emptyState.hidden = false;
  previewState.hidden = true;
  clearButton.disabled = true;
  classifyButton.disabled = true;
  classifyButton.classList.remove("loading");
  resultCard.hidden = true;
  showMessage();
}

async function runClassification() {
  if (!selectedFile) return;

  classifyButton.disabled = true;
  clearButton.disabled = true;
  classifyButton.classList.add("loading");
  resultCard.hidden = true;
  showMessage("Running the local model…");

  const formData = new FormData();
  formData.append("image", selectedFile, selectedFile.name);

  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Classification failed.");
    }

    const details = LABEL_DETAILS[payload.label];
    if (!details) throw new Error("The server returned an unknown class.");
    resultLabel.textContent = details.label;
    resultSymbol.textContent = details.symbol;
    resultCard.hidden = false;
    const source = payload.source_size;
    showMessage(
      source
        ? `${source.width}×${source.height} fitted to 224×224 for classification.`
        : "",
    );
  } catch (error) {
    showMessage(error.message || "Classification failed.");
  } finally {
    classifyButton.disabled = false;
    clearButton.disabled = false;
    classifyButton.classList.remove("loading");
  }
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
clearButton.addEventListener("click", clearSelection);
classifyButton.addEventListener("click", runClassification);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  selectFile(file);
});

checkServer();
