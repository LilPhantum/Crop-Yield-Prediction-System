/* =========================================================
   MAIN SCRIPT - Carousel, Form, Backend
   File: script.js
========================================================= */

/* =========================================================
   HERO CAROUSEL (FADE TRANSITION)
========================================================= */

const slides = document.querySelectorAll('.carousel-slide');
const indicators = document.querySelectorAll('.indicator');

let currentSlide = 0;
let carouselInterval = null;

function showSlide(index) {
    slides.forEach((slide, i) => {
        slide.classList.remove('slide-active');
        indicators[i].classList.remove('indicator-active');
    });
    slides[index].classList.add('slide-active');
    indicators[index].classList.add('indicator-active');
    currentSlide = index;
}

function startCarousel() {
    carouselInterval = setInterval(() => {
        const next = (currentSlide + 1) % slides.length;
        showSlide(next);
    }, 5000);
}

function resetCarousel() {
    clearInterval(carouselInterval);
    startCarousel();
}

indicators.forEach((indicator, index) => {
    indicator.addEventListener('click', () => {
        showSlide(index);
        resetCarousel();
    });
});

startCarousel();

/* =========================================================
   SIMULATION SETUP
========================================================= */

const simulationSection = document.querySelector('.simulation-section');
const simulationForm = document.querySelector('.simulation-form');
const overlay = simulationSection.querySelector('.simulation-overlay');
const cancelButton = overlay.querySelector('.secondary-button');

const resultsSection = document.querySelector('.results-section');
const aiSection = document.querySelector('.ai-recommendation-section');

let simulationController = null;

/* Initial UI state */
overlay.style.display = 'none';
resultsSection.style.display = 'none';
aiSection.style.display = 'none';

/* =========================================================
   RUN SIMULATION
========================================================= */
simulationForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    overlay.style.display = 'flex';
    simulationSection.classList.add('is-blurred');

    resultsSection.style.display = 'none';
    aiSection.style.display = 'none';

    const selectedCrops = simulationForm.querySelectorAll('input[name="crops"]:checked');
    const crops = Array.from(selectedCrops).map(cb => cb.value);
    
    const country = simulationForm.querySelector('[name="country"]').value || "";
    const landSize = parseFloat(simulationForm.querySelector('[name="land-size"]').value) || 0;

    // Validation
    if (crops.length === 0) {
        alert('Please select at least one crop');
        overlay.style.display = 'none';
        simulationSection.classList.remove('is-blurred');
        return;
    }
    
    if (!country) {
        alert('Please select a country');
        overlay.style.display = 'none';
        simulationSection.classList.remove('is-blurred');
        return;
    }
    
    if (landSize <= 0) {
        alert('Please enter a valid land size');
        overlay.style.display = 'none';
        simulationSection.classList.remove('is-blurred');
        return;
    }

    const payload = {
        crops: crops,
        country: country,
        land_size: landSize
    };

    simulationController = new AbortController();

    try {
        const response = await fetch("http://127.0.0.1:3000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: simulationController.signal
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();

        overlay.style.display = 'none';
        simulationSection.classList.remove('is-blurred');

        resultsSection.style.display = 'block';
        aiSection.style.display = 'block';

        // Call chart functions (from charts.js)
        renderClimateChart(data.climate, country);
        renderYieldChart(data.results);

        // Show yield details list
        renderYieldDetails(data.results);

        // AI Recommendation
        renderAIRecommendation(data, country, landSize);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error(error);
            alert(`Simulation failed: ${error.message}\n\nEnsure the backend server is running on http://localhost:3000`);
        }
        overlay.style.display = 'none';
        simulationSection.classList.remove('is-blurred');
    } finally {
        simulationController = null;
    }
});

/* =========================================================
   RENDER YIELD DETAILS (LIST)
========================================================= */
function renderYieldDetails(results) {
    const detailsContainer = document.querySelector('.yield-details-list');
    if (!detailsContainer) return;

    detailsContainer.innerHTML = '';

    results.forEach((result, index) => {
        const colors = [
            '#00E396', '#008FFB', '#FEB019', '#775DD0', 
            '#FF4560', '#00D9E9', '#FF6178', '#546E7A'
        ];
        
        const item = document.createElement('div');
        item.className = 'yield-detail-item';
        item.style.marginBottom = '1rem';
        item.style.padding = '1.25rem';
        item.style.background = '#ffffff';
        item.style.border = `2px solid ${colors[index % colors.length]}`;
        item.style.borderRadius = '10px';
        item.style.boxShadow = '0 2px 8px rgba(0,0,0,0.05)';
        
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <strong style="font-size: 1.1rem; color: ${colors[index % colors.length]};">${result.crop}</strong>
                <span style="background: ${colors[index % colors.length]}20; color: ${colors[index % colors.length]}; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">PREDICTED</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <p style="font-size: 0.85rem; color: #666; margin-bottom: 0.25rem;">Yield per Hectare</p>
                    <p style="font-size: 1.25rem; font-weight: bold; color: #1a1a1a;">${result.yield_per_ha.toLocaleString()} hg/ha</p>
                </div>
                <div>
                    <p style="font-size: 0.85rem; color: #666; margin-bottom: 0.25rem;">Total Production</p>
                    <p style="font-size: 1.25rem; font-weight: bold; color: #1a1a1a;">${result.total_production.toLocaleString()} hg</p>
                </div>
            </div>
        `;
        
        detailsContainer.appendChild(item);
    });
}

/* =========================================================
   RENDER AI RECOMMENDATION
========================================================= */
function renderAIRecommendation(data, country, landSize) {
    const aiContainer = aiSection.querySelector('.ai-output');
    
    const countryName = country.charAt(0).toUpperCase() + country.slice(1).replace('_', ' ');
    const cropsList = data.results.map(r => r.crop).join(', ');
    
    aiContainer.innerHTML = `
        <div style="line-height: 1.8;">
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                ${data.recommendation}
            </p>
            <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.2); margin: 1.5rem 0;">
            <div style="display: grid; gap: 0.75rem;">
                <p><strong>📍 Location:</strong> ${countryName}, West Africa</p>
                <p><strong>🌾 Selected Crops:</strong> ${cropsList}</p>
                <p><strong>📏 Land Size:</strong> ${landSize} hectares</p>
                <p><strong>📅 Projection Year:</strong> 2026</p>
            </div>
        </div>
    `;
}

/* =========================================================
   CANCEL SIMULATION
========================================================= */
cancelButton.addEventListener('click', () => {
    if (simulationController) {
        simulationController.abort();
    }

    overlay.style.display = 'none';
    simulationSection.classList.remove('is-blurred');
    resultsSection.style.display = 'none';
    aiSection.style.display = 'none';
    simulationForm.reset();
});

/* =========================================================
   DROPDOWN (MULTI-CHECKBOX)
========================================================= */
document.querySelectorAll(".checkbox-dropdown").forEach(dropdown => {
    const header = dropdown.querySelector(".dropdown-header");

    header.addEventListener("click", () => {
        dropdown.classList.toggle("active");
    });

    document.addEventListener("click", (e) => {
        if (!dropdown.contains(e.target)) {
            dropdown.classList.remove("active");
        }
    });
});

/* =========================================================
   ACCESSIBILITY: ESC KEY TO CANCEL
========================================================= */
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.style.display === 'flex') {
        cancelButton.click();
    }
});