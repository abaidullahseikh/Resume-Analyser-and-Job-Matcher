(function () {
  "use strict";

  const dataNode = document.getElementById("dashboard-data");
  if (!dataNode) return;
  let data;
  try {
    data = JSON.parse(dataNode.textContent || "{}");
  } catch (e) {
    return;
  }
  const breakdown = data.match_breakdown || {};

  const labels = {
    skill_match:      "Skills",
    experience_match: "Experience",
    domain_match:     "Domain",
    trajectory_match: "Trajectory",
  };
  const order = ["skill_match", "experience_match", "domain_match", "trajectory_match"];
  const present = order.filter((k) => k in breakdown);
  const labelArr = present.map((k) => labels[k]);
  const valueArr = present.map((k) => breakdown[k]);

  const accent       = "#4f46e5";
  const accentSoft   = "rgba(79, 70, 229, 0.15)";
  const gridColor    = "rgba(15, 23, 42, 0.08)";
  const tickColor    = "#64748b";

  const barColorMap = {
    skill_match:      null,   // score-based
    experience_match: null,   // score-based
    domain_match:     "#0891b2", // teal
    trajectory_match: "#9333ea", // purple
  };

  if (window.Chart) {
    const barCanvas = document.getElementById("matchBreakdownChart");
    if (barCanvas) {
      new Chart(barCanvas, {
        type: "bar",
        data: {
          labels: labelArr,
          datasets: [{
            label: "Score",
            data: valueArr,
            backgroundColor: present.map((k, i) =>
              barColorMap[k]
                ? barColorMap[k]
                : (valueArr[i] >= 80 ? "#15803d" : valueArr[i] >= 60 ? accent : valueArr[i] >= 40 ? "#b45309" : "#b91c1c")
            ),
            borderRadius: 6,
            maxBarThickness: 36,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, ticks: { color: tickColor } },
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: gridColor },
              ticks: { color: tickColor, stepSize: 25 },
            },
          },
        },
      });
    }

    const radarCanvas = document.getElementById("matchRadarChart");
    if (radarCanvas) {
      new Chart(radarCanvas, {
        type: "radar",
        data: {
          labels: labelArr,
          datasets: [{
            label: "Match",
            data: valueArr,
            backgroundColor: accentSoft,
            borderColor: accent,
            borderWidth: 2,
            pointBackgroundColor: present.map((k) => barColorMap[k] || accent),
            pointBorderColor: "#fff",
            pointHoverBackgroundColor: "#fff",
            pointHoverBorderColor: present.map((k) => barColorMap[k] || accent),
            pointRadius: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            r: {
              suggestedMin: 0,
              suggestedMax: 100,
              angleLines: { color: gridColor },
              grid: { color: gridColor },
              pointLabels: { color: tickColor, font: { size: 11, weight: "600" } },
              ticks: {
                color: tickColor,
                backdropColor: "transparent",
                stepSize: 25,
                showLabelBackdrop: false,
              },
            },
          },
        },
      });
    }
  }


  const donut1 = document.getElementById("donutMatchType");
  if (donut1 && window.Chart) {
    new Chart(donut1, {
      type: "doughnut",
      data: {
        labels: ["Direct", "Inferred", "Missing"],
        datasets: [{
          data: [data.n_direct || 0, data.n_inferred || 0, data.n_missing || 0],
          backgroundColor: ["#16a34a", "#b45309", "#dc2626"],
          borderWidth: 2,
          borderColor: "#fff",
          hoverOffset: 6,
        }],
      },
      options: {
        responsive: false,
        cutout: "68%",
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.label}: ${ctx.parsed}`,
            },
          },
        },
      },
    });
  }


  const donut2 = document.getElementById("donutReqType");
  if (donut2 && window.Chart) {
    const reqDirect   = data.n_required_direct   || 0;
    const reqInferred = data.n_required_inferred  || 0;
    const reqMissing  = data.n_required_missing   || 0;
    const optional    = data.n_optional           || 0;
    new Chart(donut2, {
      type: "doughnut",
      data: {
        labels: ["Req. met", "Req. inferred", "Req. missing", "Optional"],
        datasets: [{
          data: [reqDirect, reqInferred, reqMissing, optional],
          backgroundColor: ["#16a34a", "#f59e0b", "#dc2626", "#6366f1"],
          borderWidth: 2,
          borderColor: "#fff",
          hoverOffset: 6,
        }],
      },
      options: {
        responsive: false,
        cutout: "68%",
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.label}: ${ctx.parsed}`,
            },
          },
        },
      },
    });
  }

  const printBtn = document.getElementById("print-report-btn");
  if (printBtn) {
    printBtn.addEventListener("click", function () {
      window.print();
    });
  }
})();
