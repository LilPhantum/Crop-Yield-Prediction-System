/* =========================================================
   CHARTS SCRIPT - ApexCharts Visualizations
   File: charts.js
   Requires: ApexCharts library
========================================================= */

let climateChart = null;
let yieldChart = null;

/* =========================================================
   CLIMATE CHART (AREA CHART - RAINFALL & TEMPERATURE)
========================================================= */
function renderClimateChart(climate, country) {
    const chartContainer = document.querySelector('#climate-chart');
    
    if (!chartContainer) {
        console.error('Climate chart container not found');
        return;
    }

    // Destroy existing chart
    if (climateChart) {
        climateChart.destroy();
    }

    // Generate monthly data (simplified - using annual averages)
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const monthlyRainfall = generateMonthlyRainfall(climate.rainfall);
    const monthlyTemp = generateMonthlyTemperature(climate.temp);

    const options = {
        chart: {
            type: "area",
            height: 350,
            width: '100%',
            fontFamily: 'Inter, sans-serif',
            stacked: false,
            toolbar: {
                show: true
            },
            dropShadow: {
                enabled: true,
                enabledOnSeries: [0],
                top: -2,
                left: 2,
                blur: 5,
                opacity: 0.06
            }
        },
        colors: ['#00E396', '#0090FF'],
        stroke: {
            curve: "smooth",
            width: 3
        },
        dataLabels: {
            enabled: false
        },
        series: [{
            name: 'Rainfall (mm)',
            data: monthlyRainfall
        }, {
            name: 'Temperature (°C)',
            data: monthlyTemp
        }],
        markers: {
            size: 0,
            strokeColor: "#fff",
            strokeWidth: 3,
            strokeOpacity: 1,
            fillOpacity: 1,
            hover: {
                size: 6
            }
        },
        xaxis: {
            categories: months,
            axisBorder: {
                show: false
            },
            axisTicks: {
                show: false
            }
        },
        yaxis: [
            {
                title: {
                    text: 'Rainfall (mm)'
                },
                labels: {
                    formatter: function(val) {
                        return val.toFixed(0);
                    }
                }
            },
            {
                opposite: true,
                title: {
                    text: 'Temperature (°C)'
                },
                labels: {
                    formatter: function(val) {
                        return val.toFixed(1);
                    }
                }
            }
        ],
        grid: {
            padding: {
                left: 10,
                right: 10
            }
        },
        tooltip: {
            shared: true,
            intersect: false,
            y: [{
                formatter: function(val) {
                    return val.toFixed(0) + ' mm';
                }
            }, {
                formatter: function(val) {
                    return val.toFixed(1) + ' °C';
                }
            }]
        },
        legend: {
            position: 'top',
            horizontalAlign: 'left',
            offsetY: 0
        },
        fill: {
            type: "gradient",
            gradient: {
                shade: 'light',
                type: "vertical",
                shadeIntensity: 0.5,
                opacityFrom: 0.7,
                opacityTo: 0.3
            }
        },
        title: {
            text: `Climate Pattern - ${country.charAt(0).toUpperCase() + country.slice(1).replace('_', ' ')}`,
            align: 'center',
            style: {
                fontSize: '16px',
                fontWeight: 600,
                color: '#1f7a5f'
            }
        }
    };

    climateChart = new ApexCharts(chartContainer, options);
    climateChart.render();
}

/* =========================================================
   YIELD CHART (COLUMN CHART - DISTRIBUTED)
========================================================= */
function renderYieldChart(results) {
    const chartContainer = document.querySelector('#yield-chart');
    
    if (!chartContainer) {
        console.error('Yield chart container not found');
        return;
    }

    // Destroy existing chart
    if (yieldChart) {
        yieldChart.destroy();
    }

    const cropNames = results.map(r => r.crop);
    const yieldData = results.map(r => r.yield_per_ha);
    
    const colors = [
        '#00E396', '#008FFB', '#FEB019', '#775DD0', 
        '#FF4560', '#00D9E9', '#FF6178', '#546E7A'
    ];

    const options = {
        series: [{
            name: 'Yield (hg/ha)',
            data: yieldData
        }],
        chart: {
            height: 400,
            type: 'bar',
            fontFamily: 'Inter, sans-serif',
            toolbar: {
                show: true
            }
        },
        colors: colors.slice(0, results.length),
        plotOptions: {
            bar: {
                columnWidth: '50%',
                distributed: true,
                borderRadius: 8,
                dataLabels: {
                    position: 'top'
                }
            }
        },
        dataLabels: {
            enabled: true,
            formatter: function(val) {
                return val.toLocaleString();
            },
            offsetY: -20,
            style: {
                fontSize: '12px',
                fontWeight: 600,
                colors: ["#304758"]
            }
        },
        legend: {
            show: false
        },
        xaxis: {
            categories: cropNames,
            labels: {
                style: {
                    colors: colors.slice(0, results.length),
                    fontSize: '13px',
                    fontWeight: 600
                }
            }
        },
        yaxis: {
            title: {
                text: 'Yield (hg/ha)'
            },
            labels: {
                formatter: function(val) {
                    return val.toLocaleString();
                }
            }
        },
        grid: {
            borderColor: '#e0e0e0',
            strokeDashArray: 3
        },
        title: {
            text: 'Predicted Crop Yields (2026)',
            align: 'center',
            style: {
                fontSize: '16px',
                fontWeight: 600,
                color: '#1f7a5f'
            }
        },
        tooltip: {
            y: {
                formatter: function(val) {
                    return val.toLocaleString() + ' hg/ha';
                }
            }
        }
    };

    yieldChart = new ApexCharts(chartContainer, options);
    yieldChart.render();
}

/* =========================================================
   HELPER: GENERATE MONTHLY RAINFALL
========================================================= */
function generateMonthlyRainfall(annualTotal) {
    // Simulate West African rainfall pattern (dry season Jan-Mar, wet May-Oct)
    const pattern = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.18, 0.16, 0.12, 0.06, 0.02, 0.01];
    return pattern.map(factor => annualTotal * factor);
}

/* =========================================================
   HELPER: GENERATE MONTHLY TEMPERATURE
========================================================= */
function generateMonthlyTemperature(annualAvg) {
    // Simulate temperature variation (+/- 3°C from average)
    const variation = [-1, -2, 0, 1, 2, 3, 2, 1, 0, -1, -2, -1];
    return variation.map(diff => annualAvg + diff);
}