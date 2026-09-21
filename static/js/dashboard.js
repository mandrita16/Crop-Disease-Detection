async function loadDashboard() {
    const response = await fetch("/history?limit=50");
    const records = await response.json();

    renderStats(records);
    renderHistoryTable(records);
    renderDiseaseChart(records);
    renderQualityChart(records);
}

function renderStats(records) {
    const total = records.length;
    const needsReview = records.filter(r => r.needs_review).length;
    const goodCount = records.filter(r => r.fusion_prediction === "Good").length;
    const poorCount = records.filter(r => r.fusion_prediction === "Poor").length;

    document.getElementById("statsRow").innerHTML = `
        <div class="stat-card"><div class="num">${total}</div><div class="label">Total Predictions</div></div>
        <div class="stat-card"><div class="num">${goodCount}</div><div class="label">Good Verdicts</div></div>
        <div class="stat-card"><div class="num">${poorCount}</div><div class="label">Poor Verdicts</div></div>
        <div class="stat-card"><div class="num">${needsReview}</div><div class="label">Flagged for Review</div></div>
    `;
}

function renderHistoryTable(records) {
    const tbody = document.getElementById("historyBody");
    tbody.innerHTML = records.map(r => `
        <tr>
            <td>${r.id}</td>
            <td>${new Date(r.timestamp).toLocaleString()}</td>
            <td>${r.disease_prediction ?? "-"}</td>
            <td>${r.quality_prediction ?? "-"}</td>
            <td>${r.fusion_prediction ?? "-"}</td>
            <td class="${r.needs_review ? 'review-yes' : 'review-no'}">${r.needs_review ? "Yes" : "No"}</td>
        </tr>
    `).join("");
}

function countBy(records, field) {
    const counts = {};
    records.forEach(r => {
        const key = r[field] ?? "Unknown";
        counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
}

function renderDiseaseChart(records) {
    const counts = countBy(records, "disease_prediction");
    new Chart(document.getElementById("diseaseChart"), {
        type: "bar",
        data: {
            labels: Object.keys(counts),
            datasets: [{ label: "Count", data: Object.values(counts), backgroundColor: "#66bb6a" }],
        },
        options: { plugins: { legend: { display: false } }, indexAxis: "y" },
    });
}

function renderQualityChart(records) {
    const counts = countBy(records, "quality_prediction");
    new Chart(document.getElementById("qualityChart"), {
        type: "doughnut",
        data: {
            labels: Object.keys(counts),
            datasets: [{
                data: Object.values(counts),
                backgroundColor: ["#66bb6a", "#ffca28", "#ef5350"],
            }],
        },
    });
}

loadDashboard();
