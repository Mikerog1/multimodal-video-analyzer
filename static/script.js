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
            modelIdGroup.classList.remove('hidden');
            modelIdSelect.disabled = false;
        } else {
            modelIdGroup.classList.add('hidden');
            modelIdSelect.disabled = true;
        }
    };
    modelTypeSelect.addEventListener('change', syncModelFields);
    syncModelFields(); // Run on startup

    // QA Option Toggle
    const generateQaCheckbox = document.getElementById('generate_qa');
    const qaCategoriesGroup = document.getElementById('qa_categories_group');
    const qaCheckboxes = qaCategoriesGroup.querySelectorAll('input[type="checkbox"]');

    const syncQaFields = () => {
        if (generateQaCheckbox.checked) {
            qaCategoriesGroup.classList.remove('hidden');
            qaCheckboxes.forEach(cb => cb.disabled = false);
        } else {
            qaCategoriesGroup.classList.add('hidden');
            qaCheckboxes.forEach(cb => cb.disabled = true);
        }
    };
    generateQaCheckbox.addEventListener('change', syncQaFields);
    syncQaFields(); // Run on startup

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
        formData.set('generate_qa', document.getElementById('generate_qa').checked ? 'true' : 'false');

        // Compile QA Categories
        const activeQaCategories = [];
        if (document.getElementById('qa_counting').checked) activeQaCategories.push('counting');
        if (document.getElementById('qa_negative').checked) activeQaCategories.push('negative');
        if (document.getElementById('qa_ambiguity').checked) activeQaCategories.push('ambiguity');
        if (document.getElementById('qa_day_night').checked) activeQaCategories.push('day_night');
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
                showResults(data.results);
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

    // Display functions
    function showResults(results) {
        loadingOverlay.classList.add('hidden');
        resultsPanel.classList.remove('hidden');
        
        const videoPlayer = document.getElementById('result-video');
        const downloadCsv = document.getElementById('download-csv');
        const downloadJson = document.getElementById('download-json');
        const downloadQa = document.getElementById('download-qa');
        const downloadVideo = document.getElementById('download-video');
        
        if (results.video) {
            videoPlayer.src = results.video;
            videoPlayer.load();
            downloadVideo.href = results.video;
            downloadVideo.style.display = 'flex';
        } else {
            videoPlayer.style.display = 'none';
            downloadVideo.style.display = 'none';
        }
        
        if (results.csv) {
            downloadCsv.href = results.csv;
            downloadCsv.style.display = 'flex';
        } else {
            downloadCsv.style.display = 'none';
        }
        
        if (results.json) {
            downloadJson.href = results.json;
            downloadJson.style.display = 'flex';
        } else {
            downloadJson.style.display = 'none';
        }
        
        if (results.qa_json) {
            downloadQa.href = results.qa_json;
            downloadQa.style.display = 'flex';
        } else {
            downloadQa.style.display = 'none';
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
        syncQaFields();

        // Reset progress
        document.getElementById('progress-container').classList.add('hidden');
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('loading-spinner').style.display = 'block';

        // Reset download links visibility
        document.getElementById('download-qa').style.display = 'none';

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
                    showResults(data.results);
                    fileNameDisplay.textContent = videoParam;
                    mainPanel.classList.add('hidden');
                }
            })
            .catch(err => {
                console.log('No preloaded results found:', err.message);
            });
    }
});
