// Renders the Chart.js visualizations on the dashboard page.
// Expects importanceLabels, importanceValues, distributionData, tutoringData
// to be defined as globals in dashboard.html before this script loads.

document.addEventListener("DOMContentLoaded", () => {
  const tealSolid = "#00C2A8";
  const amberSolid = "#FFB020";
  const violetSolid = "#6C63FF";
  const navy = "#10172A";
  const gridColor = "rgba(16,23,42,0.06)";

  Chart.defaults.font.family = "Inter, sans-serif";
  Chart.defaults.color = "#5B6584";

  // ---- Feature Importance (horizontal bar) ----
  const importanceCtx = document.getElementById("importanceChart");
  if (importanceCtx && typeof importanceLabels !== "undefined") {
    new Chart(importanceCtx, {
      type: "bar",
      data: {
        labels: importanceLabels,
        datasets: [{
          data: importanceValues.map(v => (v * 100).toFixed(1)),
          backgroundColor: tealSolid,
          borderRadius: 6,
          maxBarThickness: 22,
        }]
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: gridColor }, ticks: { callback: v => v + "%" } },
          y: { grid: { display: false } }
        }
      }
    });
  }

  // ---- Score Distribution (bar histogram) ----
  const distCtx = document.getElementById("distributionChart");
  if (distCtx && typeof distributionData !== "undefined") {
    const bins = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"];
    new Chart(distCtx, {
      type: "bar",
      data: {
        labels: bins,
        datasets: [{
          label: "Students",
          data: distributionData,
          backgroundColor: violetSolid,
          borderRadius: 6,
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { grid: { color: gridColor }, beginAtZero: true },
          x: { grid: { display: false } }
        }
      }
    });
  }

  // ---- Tutoring Impact (doughnut-ish comparison bar) ----
  const tutoringCtx = document.getElementById("tutoringChart");
  if (tutoringCtx && typeof tutoringData !== "undefined") {
    new Chart(tutoringCtx, {
      type: "bar",
      data: {
        labels: ["Without Tutoring", "With Tutoring"],
        datasets: [{
          data: tutoringData,
          backgroundColor: [amberSolid, tealSolid],
          borderRadius: 8,
          maxBarThickness: 60,
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { grid: { color: gridColor }, beginAtZero: true, max: 100 },
          x: { grid: { display: false } }
        }
      }
    });
  }
});
