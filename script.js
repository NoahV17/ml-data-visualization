// Global variables to store data and models
let healthData = [];
let currentXCol = '';
let currentYCol = '';
let linearRegressionModel = null;
let decisionTreeModel = null;
let dataReady = false;

// Execute when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Show loading indicator
    showLoading('Loading data...');
    
    // Fetch and parse CSV data
    Papa.parse('https://raw.githubusercontent.com/NoahV17/portfolio_website/main/310/heart.csv', {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: function(results) {
            healthData = results.data.filter(row => 
                Object.values(row).every(val => val !== null && val !== undefined)
            );
            
            // Initialize the R-square table
            initializeRSquareTable();
            
            // Hide loading indicator and show success message
            hideLoading();
            showNotification('Data loaded successfully!', 'success');
            dataReady = true;
            
            // Enable form elements now that data is ready
            enableFormElements();
        },
        error: function(error) {
            console.error('Error loading data:', error);
            hideLoading();
            showNotification('Failed to load data. Please try again.', 'error');
        }
    });
    
    // Add event listeners
    document.getElementById('x-select').addEventListener('change', updateYAxisOptions);
    document.getElementById('y-select').addEventListener('change', checkFormValidity);
});

// Update Y-axis options to prevent selecting the same as X-axis
function updateYAxisOptions() {
    const xSelect = document.getElementById('x-select');
    const ySelect = document.getElementById('y-select');
    const xValue = xSelect.value;
    
    // Enable all options in Y select
    Array.from(ySelect.options).forEach(option => {
        option.disabled = false;
    });
    
    // Disable the option that matches X select
    if (xValue) {
        const matchingYOption = Array.from(ySelect.options).find(option => option.value === xValue);
        if (matchingYOption) matchingYOption.disabled = true;
        
        // If currently selected Y is now disabled, reset Y selection
        if (ySelect.value === xValue) {
            ySelect.value = "";
        }
    }
    
    checkFormValidity();
}

// Check if form is valid for training
function checkFormValidity() {
    const xValue = document.getElementById('x-select').value;
    const yValue = document.getElementById('y-select').value;
    const trainButton = document.querySelector('button');
    
    trainButton.disabled = !(xValue && yValue && xValue !== yValue);
    return xValue && yValue && xValue !== yValue;
}

// Enable form elements once data is loaded
function enableFormElements() {
    document.getElementById('x-select').disabled = false;
    document.getElementById('y-select').disabled = false;
    document.getElementById('XpredictY').disabled = false;
    document.querySelector('button').disabled = !checkFormValidity();
    document.getElementById('maxDepth').disabled = false;
}

// Main function to train model and visualize results
function trainAndOutput() {
    if (!dataReady) {
        showNotification('Data is not fully loaded yet. Please wait.', 'warning');
        return;
    }
    
    // Get selected columns
    const xCol = document.getElementById('x-select').value;
    const yCol = document.getElementById('y-select').value;
    
    if (!xCol || !yCol) {
        showNotification('Please select both X and Y variables', 'warning');
        return;
    }
    
    showLoading('Training model...');
    
    // Store current selections
    currentXCol = xCol;
    currentYCol = yCol;
    
    // Extract data for selected columns
    const xyData = extractXYData(xCol, yCol);
    if (!xyData || xyData.x.length === 0) {
        hideLoading();
        showNotification('No valid data found for selected columns', 'error');
        return;
    }
    
    // Perform linear regression
    try {
        const { model, xArray, yArray, rSquare, equation } = performLinearRegression(xyData);
        linearRegressionModel = model;
        
        // Create scatter plot with regression line
        createScatterPlot(xArray, yArray, xCol, yCol, model, rSquare);
        
        // Display regression statistics
        displayRegressionResults(rSquare, equation);
        
        // Make prediction if input is provided
        const predictionInput = document.getElementById('XpredictY').value;
        if (predictionInput && !isNaN(parseFloat(predictionInput))) {
            makePrediction(parseFloat(predictionInput), xCol, yCol);
        }
        
        // Create additional visualizations
        createBoxPlot(xArray, yArray, xCol, yCol);
        createHistograms(xArray, yArray, xCol, yCol);
        
        hideLoading();
        showNotification('Model trained successfully!', 'success');
    } catch (error) {
        console.error('Error in training model:', error);
        hideLoading();
        showNotification('Error training model: ' + error.message, 'error');
    }
}

// Extract X and Y data from the dataset
function extractXYData(xCol, yCol) {
    const x = [];
    const y = [];
    
    healthData.forEach(row => {
        if (row[xCol] !== undefined && row[yCol] !== undefined && 
            !isNaN(row[xCol]) && !isNaN(row[yCol])) {
            x.push(row[xCol]);
            y.push(row[yCol]);
        }
    });
    
    return { x, y };
}

// Perform linear regression analysis
function performLinearRegression(xyData) {
    const xArray = xyData.x;
    const yArray = xyData.y;
    
    try {
        // Convert data to format required by ML.js
        const data = xArray.map((x, i) => [x, yArray[i]]);
        
        // Create and train linear regression model
        // Fix: ML.LinearRegression is not a constructor error
        // Use SimpleLinearRegression from ML.js instead
        const model = new ML.SimpleLinearRegression(xArray, yArray);
        
        // Calculate R-squared manually since it's not built into ML.js
        const predictions = xArray.map(x => model.predict(x));
        const rSquare = calculateRSquared(yArray, predictions);
        
        // Generate equation using the model's parameters
        const equation = `y = ${model.slope.toFixed(4)}x + ${model.intercept.toFixed(4)}`;
        
        return { model, xArray, yArray, rSquare, equation };
    } catch (error) {
        console.error("Error in linear regression calculation:", error);
        throw new Error("Failed to perform linear regression. Check if your data has sufficient variation.");
    }
}

// Calculate R-squared value
function calculateRSquared(actual, predicted) {
    const mean = actual.reduce((sum, val) => sum + val, 0) / actual.length;
    const ssTotal = actual.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
    const ssResidual = actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0);
    return 1 - (ssResidual / ssTotal);
}

// Create scatter plot with regression line
function createScatterPlot(xArray, yArray, xCol, yCol, model, rSquare) {
    // Generate points for regression line
    const minX = Math.min(...xArray);
    const maxX = Math.max(...xArray);
    const lineX = [minX, maxX];
    const lineY = lineX.map(x => model.predict(x));
    
    // Create scatter plot
    const trace1 = {
        x: xArray,
        y: yArray,
        mode: 'markers',
        type: 'scatter',
        name: 'Data Points',
        marker: {
            color: 'rgba(52, 152, 219, 0.7)',
            size: 10
        }
    };
    
    // Create regression line
    const trace2 = {
        x: lineX,
        y: lineY,
        mode: 'lines',
        type: 'scatter',
        name: 'Regression Line',
        line: {
            color: 'rgba(231, 76, 60, 1)',
            width: 3
        }
    };
    
    const layout = {
        title: {
            text: `Scatter Plot: ${formatColumnName(xCol)} vs ${formatColumnName(yCol)}`,
            font: { size: 20 }
        },
        xaxis: { title: formatColumnName(xCol) },
        yaxis: { title: formatColumnName(yCol) },
        legend: { orientation: 'h', y: -0.2 },
        margin: { t: 60 },
        hovermode: 'closest',
        annotations: [{
            x: 0.95,
            y: 0.95,
            xref: 'paper',
            yref: 'paper',
            text: `R² = ${rSquare.toFixed(4)}`,
            showarrow: false,
            font: { size: 16, color: '#2c3e50' },
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: '#2c3e50',
            borderwidth: 1,
            borderpad: 4,
            align: 'center'
        }]
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    Plotly.newPlot('graph', [trace1, trace2], layout, config);
}

// Display regression results in the output area
function displayRegressionResults(rSquare, equation) {
    const outputDiv = document.getElementById('output1');
    const explanationDiv = document.getElementById('explanation');
    
    outputDiv.innerHTML = `
        <div class="results">
            <p>R-Square: ${rSquare.toFixed(4)}</p>
            <p>Linear Regression Equation: ${equation}</p>
            <p>This equation can be used to predict the Y value for any given X value.</p>
        </div>
    `;
    
    let explanation = '';
    
    if (rSquare > 0.7) {
        explanation = `This is a <span class="highlight">strong</span> correlation, indicating that ${formatColumnName(currentXCol)} is a good predictor of ${formatColumnName(currentYCol)}. About ${(rSquare * 100).toFixed(1)}% of the variation in ${formatColumnName(currentYCol)} can be explained by ${formatColumnName(currentXCol)}.`;
    } else if (rSquare > 0.5) {
        explanation = `This is a <span class="highlight">moderate</span> correlation between ${formatColumnName(currentXCol)} and ${formatColumnName(currentYCol)}. About ${(rSquare * 100).toFixed(1)}% of the variation in ${formatColumnName(currentYCol)} can be explained by ${formatColumnName(currentXCol)}.`;
    } else if (rSquare > 0.3) {
        explanation = `This is a <span class="highlight">weak</span> correlation between ${formatColumnName(currentXCol)} and ${formatColumnName(currentYCol)}. Only about ${(rSquare * 100).toFixed(1)}% of the variation in ${formatColumnName(currentYCol)} can be explained by ${formatColumnName(currentXCol)}.`;
    } else {
        explanation = `There is <span class="highlight">very little correlation</span> between ${formatColumnName(currentXCol)} and ${formatColumnName(currentYCol)}. These variables do not appear to have a strong linear relationship. Only about ${(rSquare * 100).toFixed(1)}% of the variation can be explained by this model.`;
    }
    
    explanationDiv.innerHTML = `
        <div class="info-box">
            <p><strong>Understanding R-Square:</strong></p>
            <p>R-Square is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variable. It ranges from 0 to 1:</p>
            <ul>
                <li>R-Square near 1: Strong predictive relationship (Most of the variation is explained by the model)</li>
                <li>R-Square near 0.5: Moderate predictive relationship (About half of the variation is explained)</li>
                <li>R-Square near 0: Weak or no predictive relationship (Very little of the variation is explained)</li>
            </ul>
            <p><strong>Interpretation:</strong> ${explanation}</p>
        </div>
    `;
}

// Make prediction based on user input
function makePrediction(xValue, xCol, yCol) {
    if (!linearRegressionModel) {
        showNotification('Please train the model first', 'warning');
        return;
    }
    
    try {
        // Fix prediction to use the correct method for SimpleLinearRegression
        const prediction = linearRegressionModel.predict(xValue);
        const outputDiv = document.getElementById('output2');
        
        outputDiv.innerHTML = `
            <div class="results">
                <p>For ${formatColumnName(xCol)} = ${xValue}</p>
                <p>Predicted ${formatColumnName(yCol)} = ${prediction.toFixed(4)}</p>
                <p class="info-note">This prediction is based on the linear relationship discovered between the variables.</p>
            </div>
        `;
    } catch (error) {
        showNotification('Error making prediction: ' + error.message, 'error');
    }
}

// Create box plots for X and Y variables
function createBoxPlot(xArray, yArray, xCol, yCol) {
    const trace1 = {
        y: xArray,
        type: 'box',
        name: formatColumnName(xCol),
        marker: {
            color: 'rgba(52, 152, 219, 0.7)'
        }
    };
    
    const trace2 = {
        y: yArray,
        type: 'box',
        name: formatColumnName(yCol),
        marker: {
            color: 'rgba(231, 76, 60, 0.7)'
        }
    };
    
    const layout = {
        title: 'Box Plot of Selected Variables',
        yaxis: { title: 'Value' },
        margin: { l: 50, r: 50, b: 100, t: 80, pad: 4 },
        showlegend: true
    };
    
    Plotly.newPlot('box_plot', [trace1, trace2], layout);
}

// Create histograms for X and Y variables
function createHistograms(xArray, yArray, xCol, yCol) {
    const trace1 = {
        x: xArray,
        type: 'histogram',
        name: formatColumnName(xCol),
        opacity: 0.7,
        marker: {
            color: 'rgba(52, 152, 219, 0.7)'
        }
    };
    
    const trace2 = {
        x: yArray,
        type: 'histogram',
        name: formatColumnName(yCol),
        opacity: 0.7,
        marker: {
            color: 'rgba(231, 76, 60, 0.7)'
        }
    };
    
    const layout = {
        title: 'Distribution of Selected Variables',
        xaxis: { title: 'Value' },
        yaxis: { title: 'Frequency' },
        barmode: 'overlay',
        margin: { l: 50, r: 50, b: 50, t: 80, pad: 4 },
        showlegend: true
    };
    
    Plotly.newPlot('histogram', [trace1, trace2], layout);
}

// Train decision tree model
function trainAndOutputTree() {
    if (!dataReady) {
        showNotification('Data is not fully loaded yet. Please wait.', 'warning');
        return;
    }
    
    showLoading('Training decision tree model...');
    
    setTimeout(() => {
        try {
            // Prepare data for decision tree
            const maxDepth = parseInt(document.getElementById('maxDepth').value) || 5;
            
            // Extract features and target
            const features = healthData.map(row => [
                row.age,
                row.resting_blood_pressure,
                row.serum_cholestoral,
                row.resting_electrocardiographic_results,
                row.maximum_heart_rate_achieved,
                row.oldpeak,
                row.slope_peak_exercise,
                row.number_of_major_vessels
            ]);
            
            const target = healthData.map(row => row.heart_disease === 1 ? 1 : 0);
            
            // Split data into training and testing sets
            const splitIndex = Math.floor(features.length * 0.7);
            const trainFeatures = features.slice(0, splitIndex);
            const testFeatures = features.slice(splitIndex);
            const trainTarget = target.slice(0, splitIndex);
            const testTarget = target.slice(splitIndex);
            
            // Create and train the decision tree classifier
            const dtc = new ML.DecisionTreeClassifier({
                gainFunction: 'gini',
                maxDepth: maxDepth,
                minNumSamples: 1
            });
            
            dtc.train(trainFeatures, trainTarget);
            decisionTreeModel = dtc;
            
            // Evaluate the model
            const predictions = testFeatures.map(sample => dtc.predict(sample));
            const accuracy = calculateAccuracy(predictions, testTarget);
            
            // Generate tree visualization
            const featureNames = [
                'Age', 'Resting BP', 'Cholesterol', 'ECG Results',
                'Max HR', 'Old Peak', 'Slope', 'Major Vessels'
            ];
            
            const treeVisualization = generateTreeVisualization(dtc.toJSON(), featureNames);
            
            // Display results
            document.getElementById('tree').innerHTML = treeVisualization;
            document.getElementById('tree_accuracy').innerHTML = `
                <div class="results">
                    <h3>Decision Tree Model Performance</h3>
                    <p>Accuracy on test data: ${(accuracy * 100).toFixed(2)}%</p>
                    <p>Model trained on ${trainFeatures.length} samples, tested on ${testFeatures.length} samples</p>
                    <p>Maximum tree depth: ${maxDepth}</p>
                    <div class="info-box">
                        <p>This decision tree model was trained to predict heart disease based on the available features.</p>
                        <p>An accuracy of ${(accuracy * 100).toFixed(2)}% means the model correctly classified 
                           ${(accuracy * 100).toFixed(2)}% of the test samples.</p>
                        <p>The tree visualization above shows the decision rules learned by the model.</p>
                    </div>
                </div>
            `;
            
            hideLoading();
            showNotification('Decision tree model trained successfully!', 'success');
        } catch (error) {
            console.error('Error training decision tree:', error);
            hideLoading();
            showNotification('Error training decision tree model: ' + error.message, 'error');
        }
    }, 100); // Small timeout to allow UI to update
}

// Calculate accuracy for classification models
function calculateAccuracy(predicted, actual) {
    let correct = 0;
    for (let i = 0; i < predicted.length; i++) {
        if (predicted[i] === actual[i]) correct++;
    }
    return correct / predicted.length;
}

// Generate HTML visualization of decision tree
function generateTreeVisualization(tree, featureNames, depth = 0) {
    const indent = '&nbsp;'.repeat(depth * 4);
    
    if (tree.category !== undefined) {
        const diagnosis = tree.category === 1 ? 'Heart Disease' : 'No Heart Disease';
        const confidence = ((tree.distribution[tree.category] / tree.samples) * 100).toFixed(1);
        
        return `${indent}<span style="color: #27ae60;">🍃 Class: ${diagnosis} (${tree.samples} samples, ${confidence}% confidence)</span><br>`;
    }
    
    const featureName = featureNames[tree.predictor];
    let html = `${indent}<span style="color: #3498db;">🔍 ${featureName} <= ${tree.threshold.toFixed(2)}</span><br>`;
    
    html += `${indent}<span style="color: #7f8c8d;">┣━ Yes:</span><br>`;
    html += generateTreeVisualization(tree.left, featureNames, depth + 1);
    
    html += `${indent}<span style="color: #7f8c8d;">┗━ No:</span><br>`;
    html += generateTreeVisualization(tree.right, featureNames, depth + 1);
    
    return html;
}

// Initialize R-square table with calculations
function initializeRSquareTable() {
    const columns = [
        'age', 'resting_blood_pressure', 'serum_cholestoral',
        'resting_electrocardiographic_results', 'maximum_heart_rate_achieved',
        'oldpeak', 'slope_peak_exercise', 'number_of_major_vessels'
    ];
    
    // Display a loading message for the table
    showNotification('Calculating correlation matrix...', 'info');
    
    columns.forEach(xCol => {
        columns.forEach(yCol => {
            const cellId = `rs_${xCol}_${yCol}`;
            const cellElement = document.getElementById(cellId);
            
            if (!cellElement) return;
            
            if (xCol === yCol) {
                cellElement.innerText = '1.0000';
                cellElement.title = 'Perfect correlation with itself';
                cellElement.style.backgroundColor = 'rgba(52, 152, 219, 0.9)';
            } else {
                const xyData = extractXYData(xCol, yCol);
                if (xyData.x.length > 0) {
                    try {
                        // Fix: Use SimpleLinearRegression instead
                        const model = new ML.SimpleLinearRegression(xyData.x, xyData.y);
                        
                        const predictions = xyData.x.map(x => model.predict(x));
                        const rSquare = calculateRSquared(xyData.y, predictions);
                        
                        cellElement.innerText = rSquare.toFixed(4);
                        // Add tooltip with more explanation
                        cellElement.title = `R²=${rSquare.toFixed(4)} between ${formatColumnName(xCol)} and ${formatColumnName(yCol)}`;
                        
                        // Color based on R-square value
                        const intensity = Math.min(Math.abs(rSquare) * 1.2, 0.9);
                        if (rSquare >= 0.3) {
                            cellElement.style.backgroundColor = `rgba(46, 204, 113, ${intensity})`;
                        } else if (rSquare >= 0.1) {
                            cellElement.style.backgroundColor = `rgba(241, 196, 15, ${intensity})`;
                        } else {
                            cellElement.style.backgroundColor = `rgba(231, 76, 60, ${intensity})`;
                        }
                    } catch {
                        cellElement.innerText = 'N/A';
                        cellElement.title = 'Could not calculate correlation';
                    }
                } else {
                    cellElement.innerText = 'N/A';
                    cellElement.title = 'Insufficient data';
                }
            }
        });
    });
    
    // Add explanation about the correlation matrix below the table
    const tableParent = document.querySelector('table').parentNode;
    const explanation = document.createElement('div');
    explanation.className = 'info-box';
    explanation.innerHTML = `
        <p><strong>About the Correlation Matrix:</strong></p>
        <p>This table shows the R² values between every pair of variables. R² values closer to 1.0 (green) 
        indicate stronger relationships, while values closer to 0.0 (red) indicate weaker relationships.</p>
        <p>The diagonal shows perfect correlation (1.0) because a variable always perfectly predicts itself.</p>
        <p>Hover over any cell to see the exact R² value.</p>
    `;
    tableParent.appendChild(explanation);
}

// Format column names for display
function formatColumnName(column) {
    return column
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// UI helper functions
function showLoading(message = 'Loading...') {
    let loadingElement = document.getElementById('loading-indicator');
    
    if (!loadingElement) {
        loadingElement = document.createElement('div');
        loadingElement.id = 'loading-indicator';
        loadingElement.style.position = 'fixed';
        loadingElement.style.top = '50%';
        loadingElement.style.left = '50%';
        loadingElement.style.transform = 'translate(-50%, -50%)';
        loadingElement.style.backgroundColor = 'rgba(44, 62, 80, 0.8)';
        loadingElement.style.color = 'white';
        loadingElement.style.padding = '20px';
        loadingElement.style.borderRadius = '8px';
        loadingElement.style.zIndex = '9999';
        loadingElement.style.display = 'flex';
        loadingElement.style.flexDirection = 'column';
        loadingElement.style.alignItems = 'center';
        loadingElement.style.justifyContent = 'center';
        document.body.appendChild(loadingElement);
    }
    
    loadingElement.innerHTML = `
        <div class="spinner" style="border: 4px solid rgba(255, 255, 255, 0.3); 
             border-radius: 50%; border-top: 4px solid white; 
             width: 40px; height: 40px; margin-bottom: 15px;
             animation: spin 1s linear infinite;"></div>
        <div>${message}</div>
    `;
    
    // Add animation style if not present
    if (!document.getElementById('loading-animation-style')) {
        const style = document.createElement('style');
        style.id = 'loading-animation-style';
        style.textContent = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;
        document.head.appendChild(style);
    }
    
    loadingElement.style.display = 'flex';
}

function hideLoading() {
    const loadingElement = document.getElementById('loading-indicator');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.position = 'fixed';
    notification.style.bottom = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 20px';
    notification.style.borderRadius = '4px';
    notification.style.fontWeight = 'bold';
    notification.style.maxWidth = '300px';
    notification.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
    notification.style.zIndex = '9999';
    notification.style.transition = 'opacity 0.5s ease-in-out';
    
    // Style based on notification type
    switch (type) {
        case 'success':
            notification.style.backgroundColor = 'rgba(46, 204, 113, 0.9)';
            notification.style.color = 'white';
            break;
        case 'error':
            notification.style.backgroundColor = 'rgba(231, 76, 60, 0.9)';
            notification.style.color = 'white';
            break;
        case 'warning':
            notification.style.backgroundColor = 'rgba(241, 196, 15, 0.9)';
            notification.style.color = '#333';
            break;
        default:
            notification.style.backgroundColor = 'rgba(52, 152, 219, 0.9)';
            notification.style.color = 'white';
    }
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    // Auto-hide notification after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 500);
    }, 3000);
}