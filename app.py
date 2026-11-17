import csv
import json
import os
from collections import Counter
from itertools import islice
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (64, 64)
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "eurosat.keras"
CLASSES_PATH = MODEL_DIR / "classes.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES_ID = {
    "AnnualCrop": "Lahan pertanian tahunan",
    "Forest": "Hutan",
    "HerbaceousVegetation": "Vegetasi herba",
    "Highway": "Jalan raya",
    "Industrial": "Kawasan industri",
    "Pasture": "Padang rumput",
    "PermanentCrop": "Lahan pertanian permanen",
    "Residential": "Permukiman",
    "River": "Sungai",
    "SeaLake": "Laut atau danau",
}

CLASS_DESCRIPTIONS_ID = {
    "AnnualCrop": "Lahan pertanian dengan tanaman yang ditanam dan dipanen setiap tahun.",
    "Forest": "Wilayah yang didominasi pepohonan atau hutan.",
    "HerbaceousVegetation": "Wilayah dengan vegetasi rendah seperti rumput dan tumbuhan herba.",
    "Highway": "Jalan raya dan area transportasi di sekitarnya.",
    "Industrial": "Kawasan pabrik, gudang, dan bangunan industri.",
    "Pasture": "Padang rumput yang umumnya digunakan untuk menggembala ternak.",
    "PermanentCrop": "Lahan tanaman tahunan seperti kebun buah atau kebun anggur.",
    "Residential": "Wilayah permukiman dan bangunan tempat tinggal.",
    "River": "Badan air yang mengalir melalui wilayah daratan.",
    "SeaLake": "Perairan laut atau danau.",
}


def get_gallery_images(dataset_folder: Path, class_names: list[str]) -> list[Path]:
    gallery_images = []
    for class_name in class_names:
        class_folder = dataset_folder / class_name
        if not class_folder.is_dir():
            continue
        class_images = (
            image_path
            for image_path in class_folder.iterdir()
            if image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        gallery_images.extend(islice(class_images, 5))
    return gallery_images


def get_predictions(
    model: tf.keras.Model, image: Image.Image, class_names: list[str]
) -> list[tuple[str, float]]:
    resized_image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)[np.newaxis, ...]
    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError("Jumlah keluaran model tidak cocok dengan daftar kelas.")

    top_indices = np.argsort(probabilities)[-3:][::-1]
    return [
        (class_names[index], float(probabilities[index]))
        for index in top_indices
    ]


def show_predictions(predictions: list[tuple[str, float]]) -> None:
    st.subheader("Tiga prediksi teratas")
    for class_name, confidence in predictions:
        display_name = CLASS_NAMES_ID.get(class_name, class_name)
        st.write(f"**{display_name}** — {confidence:.1%}")
        st.progress(confidence)


def build_results_csv(results: list[dict[str, object]]) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Nama gambar",
            "Prediksi 1",
            "Keyakinan 1",
            "Prediksi 2",
            "Keyakinan 2",
            "Prediksi 3",
            "Keyakinan 3",
        ]
    )

    for result in results:
        predictions = result["predictions"]
        row = [result["filename"]]
        for rank in range(3):
            if rank < len(predictions):
                class_name, confidence = predictions[rank]
                row.extend(
                    [
                        CLASS_NAMES_ID.get(class_name, class_name),
                        f"{confidence:.6f}",
                    ]
                )
            else:
                row.extend(["", ""])
        writer.writerow(row)

    return output.getvalue()


def build_model(class_count: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.25),
        ],
        name="augmentation",
    )

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def validate_dataset_folder(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise ValueError(f"Folder dataset tidak ditemukan: {folder}")

    class_folders = sorted(path for path in folder.iterdir() if path.is_dir())
    if len(class_folders) < 2:
        raise ValueError(
            "Folder dataset harus berisi subfolder kelas, misalnya "
            "EuroSAT_RGB/Forest dan EuroSAT_RGB/AnnualCrop."
        )

    empty_classes = [
        path.name
        for path in class_folders
        if sum(
            1
            for image_path in path.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        < 2
    ]
    if empty_classes:
        raise ValueError(
            "Setiap kelas perlu minimal 2 gambar. Periksa kelas: "
            + ", ".join(empty_classes)
        )

    return [path.name for path in class_folders]


def train_model(dataset_folder: Path, epochs: int, batch_size: int) -> None:
    class_names = validate_dataset_folder(dataset_folder)
    training_data = tf.keras.utils.image_dataset_from_directory(
        dataset_folder,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=42,
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        dataset_folder,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=42,
        shuffle=False,
    )

    if training_data.class_names != class_names:
        raise ValueError("Urutan kelas dataset berubah saat proses pemuatan.")

    autotune = tf.data.AUTOTUNE
    training_data = training_data.prefetch(buffer_size=autotune)
    validation_data = validation_data.prefetch(buffer_size=autotune)

    model = build_model(len(class_names))
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
        ],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model_path = MODEL_DIR / "eurosat.tmp.keras"
    model.save(temporary_model_path)
    os.replace(temporary_model_path, MODEL_PATH)

    temporary_classes_path = MODEL_DIR / "classes.tmp.json"
    temporary_classes_path.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_classes_path, CLASSES_PATH)
    load_model.clear()


@st.cache_resource
def load_model(model_path: str, modified_time: float) -> tf.keras.Model:
    del modified_time
    return tf.keras.models.load_model(model_path, compile=False)


def main() -> None:
    st.set_page_config(page_title="Klasifikasi EuroSAT", page_icon="🛰️")
    st.title("Klasifikasi Citra Satelit EuroSAT")
    st.write(
        "Analisis satu gambar atau beberapa citra satelit sekaligus, lalu lihat "
        "ringkasan kelas tutupan lahannya."
    )
    st.info(
        "Gunakan citra satelit atau gambar dari dataset EuroSAT. Foto biasa dari "
        "kamera ponsel dapat menghasilkan prediksi yang tidak akurat."
    )

    with st.sidebar:
        st.header("Latih model")
        local_dataset_path = Path(r"C:\EuroSATdata")
        default_dataset_path = (
            local_dataset_path if local_dataset_path.is_dir() else Path("data/EuroSAT")
        )
        dataset_path = st.text_input(
            "Lokasi folder RGB EuroSAT", str(default_dataset_path)
        )
        epochs = st.slider("Epoch maksimum", min_value=1, max_value=30, value=10)
        batch_size = st.selectbox("Ukuran batch", [16, 32, 64], index=1)
        start_training = st.button("Mulai pelatihan", type="primary")

    dataset_folder = Path(dataset_path).expanduser()

    if start_training:
        try:
            with st.spinner("Melatih model. Proses dapat memerlukan beberapa menit..."):
                train_model(Path(dataset_path).expanduser(), epochs, batch_size)
            st.success("Model berhasil dilatih dan disimpan di folder models/.")
        except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
            st.error(f"Pelatihan gagal: {error}")

    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        st.warning(
            "Model belum tersedia. Unduh dan ekstrak dataset RGB EuroSAT, "
            "atur lokasi folder di panel kiri, lalu mulai pelatihan."
        )
        return

    try:
        class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        if len(class_names) < 2:
            raise ValueError("File kelas model harus berisi minimal dua kelas.")
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except (OSError, ValueError, json.JSONDecodeError, tf.errors.OpError) as error:
        st.error(f"Model tidak dapat dibuka: {error}")
        return

    quiz_tab, upload_tab, analysis_tab, classes_tab = st.tabs(
        ["Tebak Citra", "Upload Gambar", "Analisis Banyak Gambar", "Kenali Kelas"]
    )

    with quiz_tab:
        st.write("Pilih contoh citra, tebak kelasnya, lalu cek jawaban dan prediksi AI.")
        gallery_images = get_gallery_images(dataset_folder, class_names)
        if not gallery_images:
            st.info(
                "Galeri belum tersedia. Pastikan lokasi dataset di panel kiri "
                "berisi subfolder kelas EuroSAT."
            )
        else:
            selected_path = st.selectbox(
                "Pilih contoh citra",
                gallery_images,
                format_func=lambda path: f"Contoh {gallery_images.index(path) + 1}",
            )
            try:
                image = Image.open(selected_path).convert("RGB")
            except (OSError, ValueError) as error:
                st.error(f"Gambar contoh tidak dapat dibuka: {error}")
            else:
                st.image(image, caption="Citra satelit dari dataset EuroSAT")
                guess = st.selectbox(
                    "Menurutmu, apa jenis tutupan lahannya?",
                    class_names,
                    format_func=lambda name: CLASS_NAMES_ID.get(name, name),
                    key="quiz_guess",
                )
                if st.button("Cek jawaban", type="primary"):
                    correct_class = selected_path.parent.name
                    if guess == correct_class:
                        st.success("Tebakanmu tepat!")
                    else:
                        correct_name = CLASS_NAMES_ID.get(correct_class, correct_class)
                        st.info(f"Belum tepat. Kelas sebenarnya adalah **{correct_name}**.")
                    try:
                        show_predictions(get_predictions(model, image, class_names))
                    except (ValueError, tf.errors.OpError) as error:
                        st.error(f"Gambar tidak dapat diprediksi: {error}")

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Pilih gambar untuk diklasifikasikan",
            type=["jpg", "jpeg", "png", "bmp"],
            key="user_image",
        )
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except (OSError, ValueError) as error:
                st.error(f"Gambar tidak dapat dibuka: {error}")
            else:
                st.image(image, caption="Gambar yang dipilih", use_container_width=True)
                try:
                    show_predictions(get_predictions(model, image, class_names))
                except (ValueError, tf.errors.OpError) as error:
                    st.error(f"Gambar tidak dapat diprediksi: {error}")

    with analysis_tab:
        st.write(
            "Unggah beberapa citra satelit sekaligus untuk melihat prediksi "
            "tiap gambar, ringkasan kelas, dan mengunduh hasilnya sebagai CSV."
        )
        uploaded_files = st.file_uploader(
            "Pilih satu atau beberapa gambar",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="batch_images",
        )

        if uploaded_files:
            results = []
            failed_files = []
            progress_bar = st.progress(0, text="Menyiapkan gambar...")
            for index, uploaded_file in enumerate(uploaded_files):
                try:
                    image_bytes = uploaded_file.getvalue()
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    predictions = get_predictions(model, image, class_names)
                    results.append(
                        {
                            "filename": uploaded_file.name,
                            "predictions": predictions,
                        }
                    )
                except (OSError, ValueError, tf.errors.OpError) as error:
                    failed_files.append(f"{uploaded_file.name}: {error}")
                progress_bar.progress(
                    (index + 1) / len(uploaded_files),
                    text=f"Menganalisis gambar {index + 1} dari {len(uploaded_files)}",
                )
            progress_bar.empty()

            for error in failed_files:
                st.warning(f"Gambar dilewati karena tidak dapat dianalisis: {error}")

            if results:
                st.subheader("Ringkasan analisis")
                st.metric("Gambar berhasil dianalisis", len(results))

                predicted_class_counts = Counter(
                    result["predictions"][0][0] for result in results
                )
                chart_data = [
                    {
                        "Kelas": CLASS_NAMES_ID.get(class_name, class_name),
                        "Jumlah gambar": count,
                    }
                    for class_name, count in predicted_class_counts.items()
                ]
                st.bar_chart(chart_data, x="Kelas", y="Jumlah gambar")

                table_rows = []
                for result in results:
                    predictions = result["predictions"]
                    table_rows.append(
                        {
                            "Nama gambar": result["filename"],
                            "Prediksi utama": CLASS_NAMES_ID.get(
                                predictions[0][0], predictions[0][0]
                            ),
                            "Keyakinan": f"{predictions[0][1]:.1%}",
                            "Prediksi kedua": (
                                CLASS_NAMES_ID.get(
                                    predictions[1][0], predictions[1][0]
                                )
                                if len(predictions) > 1
                                else "-"
                            ),
                            "Prediksi ketiga": (
                                CLASS_NAMES_ID.get(
                                    predictions[2][0], predictions[2][0]
                                )
                                if len(predictions) > 2
                                else "-"
                            ),
                        }
                    )
                st.dataframe(table_rows, hide_index=True, use_container_width=True)
                st.download_button(
                    "Unduh hasil sebagai CSV",
                    data=build_results_csv(results).encode("utf-8-sig"),
                    file_name="hasil_analisis_eurosat.csv",
                    mime="text/csv",
                )

    with classes_tab:
        st.write("Kenali jenis tutupan lahan yang dipelajari model.")
        for class_name in class_names:
            display_name = CLASS_NAMES_ID.get(class_name, class_name)
            description = CLASS_DESCRIPTIONS_ID.get(
                class_name, "Kelas tutupan lahan dari dataset EuroSAT."
            )
            st.markdown(f"**{display_name}**  \n{description}")


if __name__ == "__main__":
    main()
