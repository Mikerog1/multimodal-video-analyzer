document.addEventListener('DOMContentLoaded', () => {
    // Sliders setup
    const setupSlider = (sliderId, displayId) => {
        const slider = document.getElementById(sliderId);
        const display = document.getElementById(displayId);
        
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                display.textContent = e.target.value;
            });
        }
    };

    setupSlider('confidence', 'conf-val');
    setupSlider('fps_sample', 'fps-val');
    setupSlider('resize_factor', 'resize-val');

    // Model ID Toggle
    const modelTypeSelect = document.getElementById('model_type');
    const modelIdGroup = document.getElementById('model_id_group');
    const modelIdSelect = document.getElementById('model_id_select');

    const syncModelFields = () => {
        if (modelTypeSelect.value === 'yolo') {
            modelIdSelect.disabled = false;
            modelIdGroup.classList.remove('field-disabled');
        } else {
            modelIdSelect.disabled = true;
            modelIdGroup.classList.add('field-disabled');
        }
    };
    modelTypeSelect.addEventListener('change', syncModelFields);
    syncModelFields(); // Run on startup


    // Drag and drop setup
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-file');
    const fileNameDisplay = document.getElementById('file-name-display');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            updateFileName();
        }
    }

    fileInput.addEventListener('change', updateFileName);

    function updateFileName() {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
        } else {
            fileNameDisplay.textContent = '';
        }
    }

    // Form submission
    const form = document.getElementById('upload-form');
    const mainPanel = document.querySelector('.main-panel');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    const resultsPanel = document.getElementById('results-panel');
    const errorPanel = document.getElementById('error-panel');
    const errorMessage = document.getElementById('error-message');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please select a video file first.');
            return;
        }

        const formData = new FormData(form);
        
        // Convert checkboxes to explicit boolean values
        if (!formData.has('save_sampled_only')) {
            formData.append('save_sampled_only', 'false');
        } else {
            formData.set('save_sampled_only', 'true');
        }

        formData.set('generate_video', document.getElementById('generate_video').checked ? 'true' : 'false');
        formData.set('generate_csv', document.getElementById('generate_csv').checked ? 'true' : 'false');
        formData.set('generate_json', document.getElementById('generate_json').checked ? 'true' : 'false');

        // Compile QA Categories — generate_qa is true if any category is checked
        const activeQaCategories = [];
        if (document.getElementById('qa_counting').checked) activeQaCategories.push('counting');
        if (document.getElementById('qa_negative').checked) activeQaCategories.push('negative');
        if (document.getElementById('qa_ambiguity').checked) activeQaCategories.push('ambiguity');
        if (document.getElementById('qa_day_night').checked) activeQaCategories.push('day_night');
        formData.set('generate_qa', activeQaCategories.length > 0 ? 'true' : 'false');
        formData.set('qa_categories', activeQaCategories.join(','));

        // Show loading state
        mainPanel.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');
        resultsPanel.classList.add('hidden');
        errorPanel.classList.add('hidden');

        try {
            loadingStatus.textContent = 'Uploading...';
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            const taskId = data.task_id;

            pollStatus(taskId);

        } catch (error) {
            showError('Failed to start analysis: ' + error.message);
        }
    });

    // Polling function
    async function pollStatus(taskId) {
        try {
            const response = await fetch(`/api/status/${taskId}`);
            
            if (!response.ok) {
                throw new Error('Status check failed');
            }
            
            const data = await response.json();
            
            if (data.status === 'completed') {
                showResults(data.results, data.model_info);
            } else if (data.status === 'error') {
                showError(data.error || 'Unknown error occurred during analysis');
            } else {
                const progressContainer = document.getElementById('progress-container');
                const progressFill = document.getElementById('progress-fill');
                const loadingSpinner = document.getElementById('loading-spinner');
                
                // Update loading text based on status
                if (data.status === 'loading_model') {
                    loadingStatus.textContent = 'Loading AI Model...';
                    progressContainer.classList.add('hidden');
                    loadingSpinner.style.display = 'block';
                } else if (data.status === 'analyzing') {
                    loadingStatus.textContent = `Analyzing Video Frames (${data.progress || 0}%)`;
                    progressContainer.classList.remove('hidden');
                    loadingSpinner.style.display = 'none';
                    progressFill.style.width = `${data.progress || 0}%`;
                } else {
                    loadingStatus.textContent = 'Processing...';
                    progressContainer.classList.add('hidden');
                    loadingSpinner.style.display = 'block';
                }
                
                // Continue polling
                setTimeout(() => pollStatus(taskId), 2000);
            }
        } catch (error) {
            showError('Lost connection to server: ' + error.message);
        }
    }

    // Helper: build a /api/download URL from an output-relative path
    function downloadUrl(path) {
        return `/api/download?path=${encodeURIComponent(path)}`;
    }

    // Display functions
    function showResults(results, modelInfo) {
        loadingOverlay.classList.add('hidden');
        resultsPanel.classList.remove('hidden');

        // Set filename in the result header
        const filename = fileInput.files.length > 0 ? fileInput.files[0].name : 'Analysis Complete';
        document.getElementById('result-filename').textContent = filename;

        const metaDisplay = document.getElementById('meta-info-display');
        if (metaDisplay) {
            if (modelInfo) {
                metaDisplay.style.display = 'flex';
                metaDisplay.innerHTML = `
                    <span>Detector: <strong>${modelInfo.model_type || 'Unknown'}</strong></span>
                    <span>Model: <strong>${modelInfo.model_name || 'Unknown'}</strong></span>
                    <span>Device: <strong>${modelInfo.device || 'Unknown'}</strong></span>
                `;
            } else {
                metaDisplay.style.display = 'none';
            }
        }
        
        const videoPlayer = document.getElementById('result-video');
        const downloadCsv = document.getElementById('download-csv');
        const downloadJson = document.getElementById('download-json');
        const downloadVideo = document.getElementById('download-video');
        const analysisSection = document.getElementById('analysis-downloads').closest('.output-section');

        let hasAnalysisFiles = false;
        if (results.video) {
            hasAnalysisFiles = true;
            videoPlayer.src = results.video;
            videoPlayer.load();
            downloadVideo.href = downloadUrl(results.video);
            downloadVideo.removeAttribute('download');
            downloadVideo.style.display = 'flex';
        } else {
            videoPlayer.style.display = 'none';
            downloadVideo.style.display = 'none';
        }
        
        if (results.csv) {
            hasAnalysisFiles = true;
            downloadCsv.href = downloadUrl(results.csv);
            downloadCsv.removeAttribute('download');
            downloadCsv.style.display = 'flex';
        } else {
            downloadCsv.style.display = 'none';
        }
        
        if (results.json) {
            hasAnalysisFiles = true;
            downloadJson.href = downloadUrl(results.json);
            downloadJson.removeAttribute('download');
            downloadJson.style.display = 'flex';
        } else {
            downloadJson.style.display = 'none';
        }

        // Show or hide the Analysis Outputs section
        if (hasAnalysisFiles) {
            analysisSection.classList.remove('hidden');
        } else {
            analysisSection.classList.add('hidden');
        }


        // Render per-category QA download buttons
        const qaSection = document.getElementById('qa-output-section');
        const qaContainer = document.getElementById('qa-downloads-container');
        qaContainer.innerHTML = '';
        const qaFiles = results.qa_json_files || (results.qa_json ? [results.qa_json] : []);
        if (qaFiles.length > 0) {
            qaSection.classList.remove('hidden');
            qaFiles.forEach(filePath => {
                const match = filePath.match(/_qa_([^/]+)\.json$/);
                const label = match
                    ? 'Download QA: ' + match[1].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                    : 'Download QA Pairs';
                const svgIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>`;
                const btn = document.createElement('a');
                btn.href = downloadUrl(filePath);
                btn.className = 'secondary-btn';
                btn.innerHTML = svgIcon + ' ' + label;
                qaContainer.appendChild(btn);
            });
        } else {
            qaSection.classList.add('hidden');
        }
    }

    function showError(msg) {
        loadingOverlay.classList.add('hidden');
        errorPanel.classList.remove('hidden');
        errorMessage.textContent = msg;
    }

    // Reset buttons
    document.getElementById('reset-btn').addEventListener('click', resetApp);
    document.getElementById('error-reset-btn').addEventListener('click', resetApp);

    function resetApp() {
        form.reset();
        fileNameDisplay.textContent = '';
        document.getElementById('conf-val').textContent = '0.7';
        document.getElementById('fps-val').textContent = '1.0';
        document.getElementById('resize-val').textContent = '1.0';
        
        // Sync conditional UI states
        syncModelFields();

        // Reset progress
        document.getElementById('progress-container').classList.add('hidden');
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('loading-spinner').style.display = 'block';

        // Hide result sections
        document.getElementById('qa-downloads-container').innerHTML = '';
        document.getElementById('qa-output-section').classList.add('hidden');
        document.getElementById('analysis-downloads').closest('.output-section').classList.add('hidden');

        const metaDisplay = document.getElementById('meta-info-display');
        if (metaDisplay) {
            metaDisplay.innerHTML = '';
            metaDisplay.style.display = 'none';
        }

        errorPanel.classList.add('hidden');
        resultsPanel.classList.add('hidden');
        mainPanel.classList.remove('hidden');
    }

    // Preload results if video query parameter is passed
    const urlParams = new URLSearchParams(window.location.search);
    const videoParam = urlParams.get('video');
    if (videoParam) {
        fetch(`/api/results?video=${encodeURIComponent(videoParam)}`)
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('No preloaded results found');
            })
            .then(data => {
                if (data.status === 'completed') {
                    showResults(data.results, data.model_info);
                    fileNameDisplay.textContent = videoParam;
                    mainPanel.classList.add('hidden');
                }
            })
            .catch(err => {
                console.log('No preloaded results found:', err.message);
            });
    }
});
