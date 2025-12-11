// ==============================
// CONFIG: same-origin API
// ==============================
const API_BASE = "";

// DOM elements
const kInput = document.getElementById("k-input");
const refreshCheckbox = document.getElementById("refresh-checkbox");
const loadBtn = document.getElementById("load-btn");
const statusEl = document.getElementById("status");
const topBody = document.getElementById("top-body");
const bottomBody = document.getElementById("bottom-body");
const topKLabel = document.getElementById("top-k-label");
const bottomKLabel = document.getElementById("bottom-k-label");
const apiUrlDisplay = document.getElementById("api-url-display");

// Show base URL in footer
apiUrlDisplay.textContent = window.location.origin + "/api/top-bottom";

// Status helper
function setStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.classList.toggle("error", isError);
}

function clearTable(tbodyEl) {
    while (tbodyEl.firstChild) {
        tbodyEl.removeChild(tbodyEl.firstChild);
    }
}

function addRow(tbodyEl, displayIndex, ticker, company, rank) {
    const tr = document.createElement("tr");

    // position within the top/bottom slice (1..k)
    const tdIndex = document.createElement("td");
    tdIndex.textContent = displayIndex;
    tr.appendChild(tdIndex);

    const tdTicker = document.createElement("td");
    tdTicker.textContent = ticker;
    tr.appendChild(tdTicker);

    const tdCompany = document.createElement("td");
    tdCompany.textContent = company;
    tr.appendChild(tdCompany);

    // global rank (1 = best in S&P 500 universe)
    const tdRank = document.createElement("td");
    tdRank.textContent = rank.toString();
    tr.appendChild(tdRank);

    tbodyEl.appendChild(tr);
}

// Main function to call the API
async function loadRankings() {
    const kRaw = kInput.value.trim();
    let k = parseInt(kRaw, 10);
    if (isNaN(k) || k <= 0) {
        k = 20;
        kInput.value = "20";
    }

    const refresh = refreshCheckbox.checked ? "1" : "0";
    const url = `${API_BASE}/api/top-bottom?k=${k}&refresh=${refresh}`;

    setStatus("Loading rankings...");
    clearTable(topBody);
    clearTable(bottomBody);

    try {
        const res = await fetch(url);

        if (!res.ok) {
            throw new Error(`API error: HTTP ${res.status}`);
        }

        const data = await res.json();

        // Update labels
        topKLabel.textContent = data.k;
        bottomKLabel.textContent = data.k;

        // Fill top table (best ranks)
        data.top.forEach((row, idx) => {
            addRow(topBody, idx + 1, row.ticker, row.company, row.rank);
        });

        // Fill bottom table (worst ranks)
        data.bottom.forEach((row, idx) => {
            addRow(bottomBody, idx + 1, row.ticker, row.company, row.rank);
        });

        setStatus(
            `Loaded rankings for top and bottom ${data.k} out of ${data.n_universe} stocks.`,
            false
        );
    } catch (error) {
        console.error(error);
        setStatus("Error loading rankings. Check console for details.", true);
    }
}

// Button click
loadBtn.addEventListener("click", loadRankings);

// Auto-load on page open
document.addEventListener("DOMContentLoaded", () => {
    loadRankings();
});
