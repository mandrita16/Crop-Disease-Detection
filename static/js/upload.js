const dropZone = document.getElementById("dropZone");
const imageInput = document.getElementById("imageInput");
const previewImg = document.getElementById("previewImg");
const form = document.getElementById("analysisForm");
const spinner = document.getElementById("loadingSpinner");
const resultsDiv = document.getElementById("results");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        previewImg.src = URL.createObjectURL(file);
        previewImg.style.display = "block";
    }
});

dropZone.addEventListener("click", () => imageInput.click());
dropZone.addEventListener("dragover", (e) => e.preventDefault());
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    imageInput.files = e.dataTransfer.files;
    imageInput.dispatchEvent(new Event("change"));
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!imageInput.files[0]) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", imageInput.files[0]);
    ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"].forEach((field) => {
        formData.append(field, form[field].value);
    });

    resultsDiv.style.display = "none";
    spinner.style.display = "block";

    try {
        const response = await fetch("/predict/fusion", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error("Prediction failed");
        const data = await response.json();
        renderResults(data);
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        spinner.style.display = "none";
    }
});

function renderResults(data) {
    const reviewBadge = data.needs_review
        ? `<span class="badge-review">Needs manual review - branches disagree</span>`
        : "";

    resultsDiv.innerHTML = `
        <div class="result-card">
            <h3>Final Verdict: ${data.final_prediction} ${reviewBadge}</h3>
            <p>Confidence: ${(data.final_confidence * 100).toFixed(1)}%</p>

            <div class="side-by-side">
                <div>
                    <p><strong>Disease:</strong> ${data.image_result.predicted_class}
                    (${(data.image_result.confidence * 100).toFixed(1)}%)</p>
                    <img src="${data.image_result.gradcam_url}" alt="Grad-CAM overlay">
                </div>
                <div>
                    <p><strong>Soil Quality:</strong> ${data.tabular_result.predicted_label}
                    (${(data.tabular_result.confidence * 100).toFixed(1)}%)</p>
                    <canvas id="shapChart"></canvas>
                </div>
            </div>
        </div>
    `;

    resultsDiv.style.display = "block";
    renderShapChart(data.tabular_result.feature_contributions);

    resultsDiv.innerHTML += `<p style="text-align:center; margin-top:15px;"><a href="/dashboard">View all past predictions &rarr;</a></p>`;
}

function renderShapChart(contributions) {
    const ctx = document.getElementById("shapChart").getContext("2d");
    const labels = Object.keys(contributions);
    const values = Object.values(contributions);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "SHAP contribution",
                data: values,
                backgroundColor: values.map(v => v >= 0 ? "#66bb6a" : "#ef5350"),
            }],
        },
        options: {
            indexAxis: "y",
            plugins: { legend: { display: false } },
        },
    });
}
