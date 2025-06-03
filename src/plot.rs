use std::fs::File;
use std::io::Write;

use slut::dimension::Dimensionless;
use slut::tensor::Vector;

use crate::N;

pub fn plot_comparison<F, T>(
    trained_fn: F,
    target_fn: T,
    x_min: f64,
    x_max: f64,
    num_points: usize,
    output_file: &str,
) -> std::io::Result<()>
where
    F: Fn(f64) -> f64,
    T: Fn(f64) -> f64,
{
    let step = (x_max - x_min) / (num_points - 1) as f64;
    
    // Generate data points
    let mut trained_data = Vec::new();
    let mut target_data = Vec::new();
    let mut x_values = Vec::new();
    
    for i in 0..num_points {
        let x = x_min + i as f64 * step;
        let trained_y = trained_fn(x);
        let target_y = target_fn(x);
        
        x_values.push(x);
        trained_data.push(trained_y);
        target_data.push(target_y);
    }
    
    // Create HTML with embedded Chart.js
    let html_content = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Function Comparison</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{ 
            max-width: 1000px; 
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        canvas {{ 
            max-width: 100%; 
            height: 400px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Function Comparison: Trained vs Target</h1>
        <canvas id="chart"></canvas>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Mean Squared Error</h3>
                <p id="mse">Calculating...</p>
            </div>
            <div class="stat-box">
                <h3>Max Absolute Error</h3>
                <p id="mae">Calculating...</p>
            </div>
        </div>
    </div>

    <script>
        const xValues = {x_values:?};
        const trainedData = {trained_data:?};
        const targetData = {target_data:?};
        
        // Calculate statistics
        let mse = 0;
        let maxError = 0;
        for (let i = 0; i < trainedData.length; i++) {{
            const error = Math.abs(trainedData[i] - targetData[i]);
            const squaredError = Math.pow(trainedData[i] - targetData[i], 2);
            mse += squaredError;
            maxError = Math.max(maxError, error);
        }}
        mse /= trainedData.length;
        
        document.getElementById('mse').textContent = mse.toFixed(6);
        document.getElementById('mae').textContent = maxError.toFixed(6);
        
        const ctx = document.getElementById('chart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: xValues.map(x => x.toFixed(2)),
                datasets: [
                    {{
                        label: 'Target Function',
                        data: targetData,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    }},
                    {{
                        label: 'Trained Function',
                        data: trainedData,
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 1,
                        pointHoverRadius: 4,
                        borderDash: [5, 5]
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Function Approximation Results'
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'f(x)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'x'
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }}
            }}
        }});
    </script>
</body>
</html>
"#, x_values = x_values, trained_data = trained_data, target_data = target_data);

    let mut file = File::create(output_file)?;
    file.write_all(html_content.as_bytes())?;
    
    println!("Visualization saved to: {}", output_file);
    println!("Open the file in your web browser to view the comparison.");
    
    Ok(())
}

pub fn loss_curve(
    losses: &[f64],
    output_file: &str,
    threshold: Option<f64>,
) -> std::io::Result<()> {
    let epochs: Vec<usize> = (0..losses.len()).collect();
    
    // Calculate approximate derivative (gradient) of loss
    let mut loss_derivatives = Vec::new();
    for i in 1..losses.len() {
        let derivative = losses[i] - losses[i-1];
        loss_derivatives.push(derivative);
    }
    let derivative_epochs: Vec<usize> = (1..losses.len()).collect();
    
    // Remove outliers from derivatives using IQR method
    let mut sorted_derivatives = loss_derivatives.clone();
    sorted_derivatives.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let q1_idx = sorted_derivatives.len() / 4;
    let q3_idx = 3 * sorted_derivatives.len() / 4;
    let q1 = sorted_derivatives[q1_idx];
    let q3 = sorted_derivatives[q3_idx];
    let iqr = q3 - q1;
    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;
    
    // Clamp outliers to bounds
    let filtered_derivatives: Vec<f64> = loss_derivatives.iter()
        .map(|&d| d.max(lower_bound).min(upper_bound))
        .collect();
    
    // Also create a version that skips the first few epochs to avoid initial instability
    let skip_initial = 10.min(losses.len() / 10); // Skip first 10 epochs or 10% of data
    let stable_derivatives: Vec<f64> = filtered_derivatives.iter()
        .skip(skip_initial)
        .cloned()
        .collect();
    let stable_epochs: Vec<usize> = derivative_epochs.iter()
        .skip(skip_initial)
        .cloned()
        .collect();
    
    let threshold_value = threshold.unwrap_or(1e-5);
    
    // Create HTML with embedded Chart.js for loss curve
    let html_content = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Training Loss Curve</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .charts {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        canvas {{ 
            max-width: 100%; 
            height: 350px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Loss Analysis</h1>
        
        <div class="charts">
            <div>
                <h3>Loss Curve</h3>
                <canvas id="lossChart"></canvas>
            </div>
            <div>
                <h3>Loss Derivative (Filtered, Stable Period)</h3>
                <canvas id="derivativeChart"></canvas>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Initial Loss</h3>
                <p id="initialLoss">Calculating...</p>
            </div>
            <div class="stat-box">
                <h3>Final Loss</h3>
                <p id="finalLoss">Calculating...</p>
            </div>
            <div class="stat-box">
                <h3>Stable Avg Gradient</h3>
                <p id="avgGradient">Calculating...</p>
            </div>
            <div class="stat-box">
                <h3>Total Epochs</h3>
                <p id="totalEpochs">{epochs_count}</p>
            </div>
            <div class="stat-box">
                <h3>Outliers Filtered</h3>
                <p id="outliersFiltered">Calculating...</p>
            </div>
        </div>
    </div>

    <script>
        const epochs = {epochs:?};
        const losses = {losses:?};
        const stableEpochs = {stable_epochs:?};
        const stableDerivatives = {stable_derivatives:?};
        const allDerivatives = {filtered_derivatives:?};
        
        // Calculate statistics
        const initialLoss = losses[0];
        const finalLoss = losses[losses.length - 1];
        const totalEpochs = losses.length;
        const avgGradient = stableDerivatives.reduce((a, b) => a + b, 0) / stableDerivatives.length;
        
        // Count outliers that were filtered
        const originalDerivatives = {loss_derivatives:?};
        const outliersCount = originalDerivatives.filter((d, i) => 
            Math.abs(d - allDerivatives[i]) > 1e-10
        ).length;
        
        document.getElementById('initialLoss').textContent = initialLoss.toFixed(6);
        document.getElementById('finalLoss').textContent = finalLoss.toFixed(6);
        document.getElementById('avgGradient').textContent = avgGradient.toFixed(6);
        document.getElementById('outliersFiltered').textContent = outliersCount;
        
        // Loss curve chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        new Chart(lossCtx, {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Training Loss',
                        data: losses,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 1,
                        pointHoverRadius: 4,
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Training Loss Over Time'
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'logarithmic',
                        title: {{
                            display: true,
                            text: 'Loss (log scale)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Epoch'
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }}
            }}
        }});
        
        // Derivative chart
        const derivCtx = document.getElementById('derivativeChart').getContext('2d');
        new Chart(derivCtx, {{
            type: 'line',
            data: {{
                labels: stableEpochs,
                datasets: [
                    {{
                        label: 'Loss Change',
                        data: stableDerivatives,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 1,
                        pointHoverRadius: 4,
                        tension: 0.1
                    }},
                    {{
                        label: 'Convergence Threshold',
                        data: Array(stableEpochs.length).fill({threshold_value}),
                        borderColor: 'rgb(255, 165, 0)',
                        backgroundColor: 'rgba(255, 165, 0, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0,
                        borderDash: [10, 5]
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Loss Gradient (Δloss/Δepoch)'
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        title: {{
                            display: true,
                            text: 'Loss Change'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Epoch'
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }}
            }}
        }});
    </script>
</body>
</html>
"#, epochs = epochs, losses = losses, epochs_count = losses.len(), 
    stable_epochs = stable_epochs, stable_derivatives = stable_derivatives,
    filtered_derivatives = filtered_derivatives, loss_derivatives = loss_derivatives);

    let mut file = File::create(output_file)?;
    file.write_all(html_content.as_bytes())?;
    
    println!("Loss curve saved to: {}", output_file);
    println!("Open the file in your web browser to view the training progress.");
    
    Ok(())
}

