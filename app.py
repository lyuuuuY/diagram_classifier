"""Local Flask interface for the diagram classifier."""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from classifier import classify


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
REQUIRED_SIZE = (224, 224)
MAX_SOURCE_PIXELS = 50_000_000
VALID_LABELS = {"lewis", "graph", "equation"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.get("/")
def index() -> str:
    """Render the local upload interface."""
    return render_template("index.html")


@app.get("/api/health")
def health():
    """Small endpoint used by the page to confirm that Flask is running."""
    return jsonify({"status": "ready"})


def _prepare_classifier_image(source_image: Image.Image) -> Image.Image:
    """Proportionally fit an arbitrary PNG onto a 224x224 background."""
    if "A" in source_image.getbands():
        transparent_image = source_image.convert("RGBA")
        white_background = Image.new(
            "RGBA",
            transparent_image.size,
            (255, 255, 255, 255),
        )
        source_rgb = Image.alpha_composite(
            white_background,
            transparent_image,
        ).convert("RGB")
    else:
        source_rgb = source_image.convert("RGB")

    width, height = source_rgb.size
    corner_pixels = (
        source_rgb.getpixel((0, 0)),
        source_rgb.getpixel((width - 1, 0)),
        source_rgb.getpixel((0, height - 1)),
        source_rgb.getpixel((width - 1, height - 1)),
    )
    background_color = tuple(
        round(
            sum(pixel[channel] for pixel in corner_pixels)
            / len(corner_pixels)
        )
        for channel in range(3)
    )

    fitted_image = ImageOps.contain(
        source_rgb,
        REQUIRED_SIZE,
        method=Image.Resampling.LANCZOS,
    )
    classifier_image = Image.new(
        "RGB",
        REQUIRED_SIZE,
        background_color,
    )
    paste_position = (
        (REQUIRED_SIZE[0] - fitted_image.width) // 2,
        (REQUIRED_SIZE[1] - fitted_image.height) // 2,
    )
    classifier_image.paste(fitted_image, paste_position)
    return classifier_image


@app.post("/api/classify")
def classify_image():
    """Validate one uploaded PNG and return its predicted class."""
    uploaded_file = request.files.get("image")
    if uploaded_file is None:
        return jsonify({"error": "Please choose a PNG image."}), 400
    if not uploaded_file.filename:
        return jsonify({"error": "The uploaded file has no filename."}), 400

    image_bytes = uploaded_file.read()
    if not image_bytes:
        return jsonify({"error": "The uploaded file is empty."}), 400

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format != "PNG":
                return jsonify({"error": "Only PNG images are accepted."}), 400
            if image.width < 1 or image.height < 1:
                return jsonify({"error": "The PNG has invalid dimensions."}), 400
            if image.width * image.height > MAX_SOURCE_PIXELS:
                return (
                    jsonify(
                        {
                            "error": (
                                "The PNG is too large to process safely. "
                                "Use an image below 50 megapixels."
                            )
                        }
                    ),
                    400,
                )
            source_size = image.size
            image.load()
            classifier_image = _prepare_classifier_image(image)
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
    ):
        return jsonify({"error": "The file is not a valid PNG image."}), 400

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        classifier_image.save(temporary_path, format="PNG")

        label = classify(str(temporary_path))
        if label not in VALID_LABELS:
            raise RuntimeError(f"Unexpected classifier result: {label}")
        return jsonify(
            {
                "label": label,
                "source_size": {
                    "width": source_size[0],
                    "height": source_size[1],
                },
                "classifier_size": {
                    "width": REQUIRED_SIZE[0],
                    "height": REQUIRED_SIZE[1],
                },
            }
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error: RequestEntityTooLarge):
    return jsonify({"error": "The PNG must be smaller than 5 MB."}), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    app.logger.exception("Classification request failed", exc_info=error)
    return (
        jsonify(
            {
                "error": (
                    "The classifier could not process this image. "
                    "See the local terminal for details."
                )
            }
        ),
        500,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )
