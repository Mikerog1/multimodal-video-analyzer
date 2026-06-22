document.addEventListener('DOMContentLoaded', () => {
    const videoPlayer = document.getElementById('result-video');
    let currentQAPairs = [];

    // Listen for video metadata to render timeline markers as soon as duration is available
    if (videoPlayer) {
        videoPlayer.addEventListener('loadedmetadata', () => {
            if (currentQAPairs && currentQAPairs.length > 0) {
                updateTimelineMarkers(currentQAPairs, videoPlayer);
            }
        });
    }

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

    // Hugging Face setup
    const tabUpload = document.getElementById('tab-upload');
    const tabHuggingface = document.getElementById('tab-huggingface');
    const hfZone = document.getElementById('hf-zone');
    const hfRepoInput = document.getElementById('hf_repo_id');
    const hfTokenInput = document.getElementById('hf_token');
    const hfFetchBtn = document.getElementById('hf-fetch-btn');
    const hfFilesContainer = document.getElementById('hf-files-container');
    const hfFileInput = document.getElementById('hf_file_path');
    const hfErrorDisplay = document.getElementById('hf-error-display');
    
    let currentInputTab = 'upload';

    tabUpload.addEventListener('click', () => {
        currentInputTab = 'upload';
        tabUpload.classList.add('active');
        tabHuggingface.classList.remove('active');
        dropZone.classList.remove('hidden');
        hfZone.classList.add('hidden');
        // Clear file input
        fileInput.value = '';
        fileNameDisplay.textContent = '';
    });

    tabHuggingface.addEventListener('click', () => {
        currentInputTab = 'huggingface';
        tabHuggingface.classList.add('active');
        tabUpload.classList.remove('active');
        hfZone.classList.remove('hidden');
        dropZone.classList.add('hidden');
        // Clear file input
        fileInput.value = '';
        fileNameDisplay.textContent = '';
    });

    // Fetch video list from Hugging Face
    hfFetchBtn.addEventListener('click', async () => {
        const repoId = hfRepoInput.value.trim();
        const token = hfTokenInput.value.trim();
        
        if (!repoId) {
            showHfError('Please enter a Hugging Face Dataset link or Repository ID.');
            return;
        }

        hfFetchBtn.disabled = true;
        hfFetchBtn.textContent = 'Fetching...';
        clearHfError();
        hfFilesContainer.classList.add('hidden');

        try {
            const url = `/api/hf/list-videos?repo_id=${encodeURIComponent(repoId)}` + (token ? `&token=${encodeURIComponent(token)}` : '');
            const response = await fetch(url);
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to retrieve video files.');
            }

            const data = await response.json();
            
            if (!data.videos || data.videos.length === 0) {
                throw new Error('No video files (.mp4, .avi, .mov, .mkv, .webm) found in this repository.');
            }

            // Fill select dropdown options
            hfFileInput.innerHTML = '';
            data.videos.forEach(video => {
                const opt = document.createElement('option');
                opt.value = video;
                opt.textContent = video;
                hfFileInput.appendChild(opt);
            });

            // Set repo input value to clean ID if changed
            if (data.repo_id) {
                hfRepoInput.value = data.repo_id;
            }

            // Auto-select file if parsed from URL
            if (data.auto_selected_file) {
                const found = data.videos.find(v => v.toLowerCase() === data.auto_selected_file.toLowerCase() || v.toLowerCase().endsWith(data.auto_selected_file.toLowerCase()));
                if (found) {
                    hfFileInput.value = found;
                }
            }

            hfFilesContainer.classList.remove('hidden');
        } catch (err) {
            showHfError(err.message);
        } finally {
            hfFetchBtn.disabled = false;
            hfFetchBtn.textContent = 'Fetch Videos';
        }
    });

    // Automatically trigger Fetch Videos when user paste/types a full URL containing huggingface.co
    hfRepoInput.addEventListener('input', () => {
        const value = hfRepoInput.value.trim();
        if (value.includes('huggingface.co/')) {
            // Wait slightly for paste to complete, then fetch
            setTimeout(() => {
                if (hfRepoInput.value.trim() === value) {
                    hfFetchBtn.click();
                }
            }, 300);
        }
    });

    function showHfError(msg) {
        hfErrorDisplay.textContent = msg;
        hfErrorDisplay.classList.remove('hidden');
    }

    function clearHfError() {
        hfErrorDisplay.textContent = '';
        hfErrorDisplay.classList.add('hidden');
    }

    // ─── Sidebar ───
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebarOpenBtn = document.getElementById('sidebar-open-btn');
    const historyList = document.getElementById('history-list');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');

    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.add('collapsed');
        sidebarOpenBtn.classList.remove('hidden');
    });

    sidebarOpenBtn.addEventListener('click', () => {
        sidebar.classList.remove('collapsed');
        sidebarOpenBtn.classList.add('hidden');
    });

    // New Analysis button in sidebar — resets to upload form
    newAnalysisBtn.addEventListener('click', () => {
        resetApp();
    });

    // ─── Advanced Settings Toggle ───
    const advancedToggle = document.getElementById('advanced-settings-toggle');
    const advancedContent = document.getElementById('advanced-settings-content');

    advancedToggle.addEventListener('click', () => {
        advancedToggle.classList.toggle('open');
        advancedContent.classList.toggle('open');
    });

    // ─── Context & Guidance Toggle ───
    const guidanceToggle = document.getElementById('guidance-settings-toggle');
    const guidanceContent = document.getElementById('guidance-settings-content');

    if (guidanceToggle && guidanceContent) {
        guidanceToggle.addEventListener('click', () => {
            guidanceToggle.classList.toggle('open');
            guidanceContent.classList.toggle('open');
        });
    }

    // Currently active history folder (for highlighting)
    let activeHistoryFolder = null;

    function loadHistory() {
        fetch('/api/history')
            .then(res => res.json())
            .then(runs => {
                renderHistoryList(runs);
            })
            .catch(err => {
                console.error('Failed to load history:', err);
            });
    }

    function renderHistoryList(runs) {
        if (!runs || runs.length === 0) {
            historyList.innerHTML = `
                <div class="sidebar-empty">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.3">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                    <p>No analyses yet</p>
                </div>
            `;
            return;
        }

        historyList.innerHTML = '';
        runs.forEach(run => {
            const item = document.createElement('div');
            item.className = 'history-item';
            if (run.folder === activeHistoryFolder) {
                item.classList.add('active');
            }
            item.dataset.folder = run.folder;

            const modelBadge = run.model_info
                ? `<span class="meta-badge">${run.model_info.model_type}</span>`
                : '';

            item.innerHTML = `
                <div class="history-item-name" title="${run.video_name || run.folder}">${run.video_name || run.folder}</div>
                <div class="history-item-meta">
                    <span>${run.run_date || '—'}</span>
                    ${modelBadge}
                </div>
            `;

            item.addEventListener('click', () => {
                loadHistoryDetail(run);
                // Highlight active item
                activeHistoryFolder = run.folder;
                historyList.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
            });

            historyList.appendChild(item);
        });
    }

    function loadHistoryDetail(run) {
        // Show results from a history entry
        const results = run.files;
        const modelInfo = run.model_info;
        const analysisSettings = run.analysis_settings;

        // Translate files structure to the same format showResults expects
        const normalizedResults = {
            folder: `/output/${run.folder}`,
            video: results.video,
            is_original_video: results.is_original_video || false,
            csv: results.csv,
            json: results.json,
            qa_json_files: results.qa_json_files || [],
        };

        showResults(normalizedResults, modelInfo, run.video_name || run.folder, analysisSettings, run.object_counts);
    }

    // Form submission
    const form = document.getElementById('upload-form');
    const mainPanel = document.getElementById('main-panel');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    const resultsPanel = document.getElementById('results-panel');
    const errorPanel = document.getElementById('error-panel');
    const errorMessage = document.getElementById('error-message');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        if (currentInputTab === 'upload') {
            if (!fileInput.files.length) {
                alert('Please select a video file first.');
                return;
            }
            // Remove Hugging Face fields to keep payload clean
            formData.delete('hf_repo_id');
            formData.delete('hf_file_path');
            formData.delete('hf_token');
        } else {
            const repoId = hfRepoInput.value.trim();
            const filePath = hfFileInput.value;
            const token = hfTokenInput.value.trim();

            if (!repoId || !filePath || hfFilesContainer.classList.contains('hidden')) {
                alert('Please enter a Hugging Face dataset and select a video file first.');
                return;
            }

            // Remove file field
            formData.delete('file');
            formData.set('hf_repo_id', repoId);
            formData.set('hf_file_path', filePath);
            if (token) {
                formData.set('hf_token', token);
            } else {
                formData.delete('hf_token');
            }
        }
        
        // Convert checkboxes to explicit boolean values
        if (!formData.has('save_sampled_only')) {
            formData.append('save_sampled_only', 'false');
        } else {
            formData.set('save_sampled_only', 'true');
        }

        formData.set('generate_video', document.getElementById('generate_video').checked ? 'true' : 'false');
        formData.set('generate_csv', document.getElementById('generate_csv').checked ? 'true' : 'false');
        formData.set('generate_json', document.getElementById('generate_json').checked ? 'true' : 'false');

        // Processing options
        formData.set('remove_audio', document.getElementById('remove_audio').checked ? 'true' : 'false');
        formData.set('mask_persons', document.getElementById('mask_persons').checked ? 'true' : 'false');

        // Compile QA Categories — generate_qa is true if any category is checked
        const captionsVal = document.getElementById('captions') ? document.getElementById('captions').value.trim() : '';
        const questionsVal = document.getElementById('example_questions') ? document.getElementById('example_questions').value.trim() : '';

        const activeQaCategories = [];
        if (document.getElementById('qa_counting').checked) activeQaCategories.push('counting');
        if (document.getElementById('qa_negative').checked) activeQaCategories.push('negative');
        if (document.getElementById('qa_ambiguity').checked) activeQaCategories.push('ambiguity');
        if (document.getElementById('qa_day_night').checked) activeQaCategories.push('day_night');
        
        if (captionsVal || questionsVal) {
            activeQaCategories.push('user_queries');
        }

        formData.set('generate_qa', activeQaCategories.length > 0 ? 'true' : 'false');
        formData.set('qa_categories', activeQaCategories.join(','));

        // Show loading state
        mainPanel.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');
        resultsPanel.classList.add('hidden');
        errorPanel.classList.add('hidden');

        try {
            loadingStatus.textContent = currentInputTab === 'upload' ? 'Uploading...' : 'Starting Hugging Face Analysis...';
            
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
                showResults(data.results, data.model_info, undefined, data.analysis_settings, data.object_counts);
                // Refresh the sidebar history to include the new run
                loadHistory();
            } else if (data.status === 'error') {
                showError(data.error || 'Unknown error occurred during analysis');
            } else {
                const progressContainer = document.getElementById('progress-container');
                const progressFill = document.getElementById('progress-fill');
                const loadingSpinner = document.getElementById('loading-spinner');
                
                // Update loading text based on status
                if (data.status === 'downloading_dataset') {
                    loadingStatus.textContent = 'Downloading video from Hugging Face...';
                    progressContainer.classList.add('hidden');
                    loadingSpinner.style.display = 'block';
                } else if (data.status === 'loading_model') {
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

    // Helper: build a streaming video URL from an output-relative path
    function streamVideoUrl(path) {
        return `/api/stream-video?path=${encodeURIComponent(path)}`;
    }

    // Display functions
    function showResults(results, modelInfo, displayName, analysisSettings, objectCounts) {
        loadingOverlay.classList.add('hidden');
        mainPanel.classList.add('hidden');
        resultsPanel.classList.remove('hidden');
        errorPanel.classList.add('hidden');

        // Set filename in the result header
        const filename = displayName || (fileInput.files.length > 0 ? fileInput.files[0].name : 'Analysis Complete');
        document.getElementById('result-filename').textContent = filename;

        const metaDisplay = document.getElementById('meta-info-display');
        if (metaDisplay) {
            const hasModel = modelInfo && (modelInfo.model_type || modelInfo.model_name);
            const hasSettings = analysisSettings && Object.values(analysisSettings).some(v => v != null);
            if (hasModel || hasSettings) {
                metaDisplay.style.display = 'grid';
                let html = '';
                if (hasModel) {
                    html += `<span>Detector: <strong>${modelInfo.model_type || '—'}</strong></span>`;
                    html += `<span>Model: <strong>${modelInfo.model_name || '—'}</strong></span>`;
                }
                if (hasSettings) {
                    if (analysisSettings.resolution)
                        html += `<span>Resolution: <strong>${analysisSettings.resolution}</strong></span>`;
                    if (analysisSettings.total_frames != null)
                        html += `<span>Frames: <strong>${analysisSettings.total_frames.toLocaleString()}</strong></span>`;
                    if (analysisSettings.fps != null)
                        html += `<span>FPS: <strong>${analysisSettings.fps}</strong></span>`;
                    if (analysisSettings.duration_seconds != null) {
                        const dur = analysisSettings.duration_seconds;
                        const mins = Math.floor(dur / 60);
                        const secs = Math.round(dur % 60);
                        html += `<span>Duration: <strong>${mins}:${String(secs).padStart(2,'0')}</strong></span>`;
                    }
                    if (analysisSettings.fps_sample != null)
                        html += `<span>FPS Sampling: <strong>${analysisSettings.fps_sample}</strong></span>`;
                    if (analysisSettings.confidence_threshold != null)
                        html += `<span>Confidence: <strong>${analysisSettings.confidence_threshold}</strong></span>`;
                }
                metaDisplay.innerHTML = html;
            } else {
                metaDisplay.style.display = 'none';
            }
        }

        // Render object count stats
        const statsDisplay = document.getElementById('object-stats-display');
        if (statsDisplay) {
            statsDisplay.innerHTML = '';
            if (objectCounts && Object.keys(objectCounts).length > 0) {
                statsDisplay.classList.remove('hidden');
                
                // Map common classes to appropriate emojis
                const iconMap = {
                    'person': '🚶',
                    'car': '🚗',
                    'truck': '🚚',
                    'bus': '🚌',
                    'motorcycle': '🏍️',
                    'bicycle': '🚲',
                    'dog': '🐶',
                    'cat': '🐱',
                    'traffic light': '🚦',
                    'stop sign': '🛑',
                    'bench': '🪑',
                    'fire hydrant': '🧯',
                    'stroller': '👶'
                };
                
                Object.entries(objectCounts).forEach(([label, count]) => {
                    const icon = iconMap[label.toLowerCase()] || '📦';
                    const card = document.createElement('div');
                    card.className = 'stat-card';
                    card.innerHTML = `
                        <div class="stat-card-icon">${icon}</div>
                        <div class="stat-card-content">
                            <span class="stat-card-count">${count}</span>
                            <span class="stat-card-label">${label}</span>
                        </div>
                    `;
                    statsDisplay.appendChild(card);
                });
            } else {
                statsDisplay.classList.add('hidden');
            }
        }
        
        // Use the global videoPlayer element defined at the DOMContentLoaded scope
        const downloadCsv = document.getElementById('download-csv');
        const downloadJson = document.getElementById('download-json');
        const downloadVideo = document.getElementById('download-video');
        const analysisSection = document.getElementById('analysis-downloads').closest('.output-section');
        const videoTypeBadge = document.getElementById('video-type-badge');

        let hasAnalysisFiles = false;
        if (results.video) {
            hasAnalysisFiles = true;
            const videoContainer = videoPlayer.closest('.video-container');

            // Reset any previous error state
            const existingError = videoContainer.querySelector('.video-error-msg');
            if (existingError) existingError.remove();
            videoPlayer.style.display = 'block';

            videoPlayer.src = streamVideoUrl(results.video);
            videoPlayer.load();
            videoContainer.style.display = 'block';
            downloadVideo.href = downloadUrl(results.video);
            downloadVideo.removeAttribute('download');
            downloadVideo.style.display = 'flex';

            // Show/hide original video badge
            if (results.is_original_video) {
                videoTypeBadge.classList.remove('hidden');
            } else {
                videoTypeBadge.classList.add('hidden');
            }

            // Handle video decode errors (e.g., mp4v codec not supported by browser)
            videoPlayer.onerror = () => {
                videoPlayer.style.display = 'none';
                if (!videoContainer.querySelector('.video-error-msg')) {
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'video-error-msg';
                    errorMsg.innerHTML = `
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.5">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="15" y1="9" x2="9" y2="15"></line>
                            <line x1="9" y1="9" x2="15" y2="15"></line>
                        </svg>
                        <p>This video uses a codec (mp4v) that your browser cannot play.</p>
                        <p class="video-error-hint">Use the download button below to play it locally, or re-analyze with the <strong>avc1 (H.264)</strong> codec for browser playback.</p>
                    `;
                    videoContainer.appendChild(errorMsg);
                }
            };
        } else {
            videoPlayer.style.display = 'none';
            videoPlayer.closest('.video-container').style.display = 'none';
            downloadVideo.style.display = 'none';
            videoTypeBadge.classList.add('hidden');
        }

        // Load the analysis timeline chart
        const timelineSection = document.getElementById('analysis-timeline-section');
        if (results.folder) {
            loadAnalysisTimeline(results.folder, videoPlayer);
        } else {
            timelineSection.classList.add('hidden');
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

        // Load and display QA timeline if QA files exist
        if (qaFiles.length > 0 && results.folder) {
            loadQATimeline(results.folder, videoPlayer);
        }
    }

    // ─── Analysis Timeline Chart ───

    let timelineData = null;        // Cached timeline array from the API
    let timelineDuration = 0;       // Video duration in seconds from the timeline API
    let timelineAnimFrame = null;   // requestAnimationFrame ID for playhead updates

    async function loadAnalysisTimeline(folder, videoPlayer) {
        /**
         * Fetch detection-count timeline data and render an interactive chart.
         */
        const section = document.getElementById('analysis-timeline-section');
        const canvas = document.getElementById('analysis-timeline-canvas');
        const playhead = document.getElementById('timeline-playhead');
        const timeLabels = document.getElementById('timeline-time-labels');

        // Extract folder name from URL path
        const folderName = folder.replace(/^\/output\//, '').split('/').pop();

        try {
            const response = await fetch(`/api/analysis-timeline?folder=${encodeURIComponent(folderName)}`);
            if (!response.ok) {
                section.classList.add('hidden');
                return;
            }
            const data = await response.json();

            if (!data.timeline || data.timeline.length === 0) {
                section.classList.add('hidden');
                return;
            }

            timelineData = data.timeline;
            timelineDuration = data.duration_seconds || timelineData[timelineData.length - 1].second || 1;

            // Show the section
            section.classList.remove('hidden');

            // Render the chart
            renderTimelineChart(canvas, timelineData, timelineDuration);

            // Generate time labels
            renderTimeLabels(timeLabels, timelineDuration);

            // Click-to-seek on the chart
            const canvasWrap = canvas.closest('.analysis-timeline-canvas-wrap');
            canvasWrap.addEventListener('click', (e) => {
                const rect = canvasWrap.getBoundingClientRect();
                const xRatio = (e.clientX - rect.left) / rect.width;
                const seekTime = xRatio * timelineDuration;
                videoPlayer.currentTime = Math.max(0, Math.min(seekTime, timelineDuration));
                videoPlayer.play();
            });

            // Hover tooltip
            let tooltip = canvasWrap.querySelector('.timeline-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.className = 'timeline-tooltip';
                tooltip.style.display = 'none';
                canvasWrap.appendChild(tooltip);
            }

            canvasWrap.addEventListener('mousemove', (e) => {
                const rect = canvasWrap.getBoundingClientRect();
                const xRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const hoverSecond = Math.round(xRatio * timelineDuration);
                const entry = timelineData.find(d => d.second === hoverSecond) || timelineData[Math.min(hoverSecond, timelineData.length - 1)];

                if (entry) {
                    tooltip.style.display = 'block';
                    const tooltipX = e.clientX - rect.left;
                    tooltip.style.left = `${Math.max(50, Math.min(tooltipX, rect.width - 50))}px`;
                    tooltip.style.top = '4px';
                    tooltip.innerHTML = `
                        <div class="tt-time">${secondsToTimeString(entry.second || hoverSecond)}</div>
                        <div class="tt-row"><span class="tt-dot" style="background:#60a5fa"></span> People: ${entry.people ?? 0}</div>
                        <div class="tt-row"><span class="tt-dot" style="background:#f97316"></span> Cars: ${entry.cars ?? 0}</div>
                        <div class="tt-row"><span class="tt-dot" style="background:#a78bfa"></span> Dogs: ${entry.dogs ?? 0}</div>
                    `;
                }
            });

            canvasWrap.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });

            // Playhead: update on video timeupdate
            if (timelineAnimFrame) cancelAnimationFrame(timelineAnimFrame);

            function updatePlayhead() {
                if (!videoPlayer.paused && videoPlayer.duration && isFinite(videoPlayer.duration)) {
                    const xPercent = (videoPlayer.currentTime / timelineDuration) * 100;
                    playhead.style.left = `${Math.min(100, xPercent)}%`;
                    playhead.style.display = 'block';
                }
                timelineAnimFrame = requestAnimationFrame(updatePlayhead);
            }

            videoPlayer.addEventListener('play', () => {
                playhead.style.display = 'block';
                updatePlayhead();
            });

            videoPlayer.addEventListener('pause', () => {
                if (timelineAnimFrame) cancelAnimationFrame(timelineAnimFrame);
                // Keep playhead visible at current position
                if (videoPlayer.duration && isFinite(videoPlayer.duration)) {
                    const xPercent = (videoPlayer.currentTime / timelineDuration) * 100;
                    playhead.style.left = `${Math.min(100, xPercent)}%`;
                }
            });

            videoPlayer.addEventListener('seeked', () => {
                if (videoPlayer.duration && isFinite(videoPlayer.duration)) {
                    const xPercent = (videoPlayer.currentTime / timelineDuration) * 100;
                    playhead.style.left = `${Math.min(100, xPercent)}%`;
                    playhead.style.display = 'block';
                }
            });

        } catch (error) {
            console.error('Error loading analysis timeline:', error);
            section.classList.add('hidden');
        }
    }

    function renderTimelineChart(canvas, data, duration) {
        /**
         * Render a filled line chart on canvas showing people, cars, dogs over time.
         */
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const padTop = 8;
        const padBottom = 4;
        const chartH = h - padTop - padBottom;

        // Clear
        ctx.clearRect(0, 0, w, h);

        // Find max value for Y-axis scaling
        let maxVal = 1;
        data.forEach(d => {
            maxVal = Math.max(maxVal, d.people || 0, d.cars || 0, d.dogs || 0);
        });
        maxVal = Math.ceil(maxVal * 1.15); // Add 15% headroom

        // Draw subtle grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        for (let i = 1; i <= 4; i++) {
            const y = padTop + (chartH * (1 - i / 4));
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Helper: draw a filled area line
        function drawSeries(key, color, fillAlpha) {
            if (data.length === 0) return;

            ctx.beginPath();
            ctx.moveTo(0, padTop + chartH); // Start at bottom-left

            data.forEach((d, i) => {
                const x = (d.second / duration) * w;
                const val = d[key] || 0;
                const y = padTop + chartH - (val / maxVal) * chartH;
                if (i === 0) {
                    ctx.lineTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });

            // Close the path along the bottom
            const lastX = (data[data.length - 1].second / duration) * w;
            ctx.lineTo(lastX, padTop + chartH);
            ctx.closePath();

            // Fill
            ctx.fillStyle = color.replace(')', `, ${fillAlpha})`).replace('rgb', 'rgba');
            ctx.fill();

            // Stroke
            ctx.beginPath();
            data.forEach((d, i) => {
                const x = (d.second / duration) * w;
                const val = d[key] || 0;
                const y = padTop + chartH - (val / maxVal) * chartH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Draw series (back to front for layering)
        drawSeries('dogs', 'rgb(167, 139, 250)', 0.15);   // Purple
        drawSeries('cars', 'rgb(249, 115, 22)', 0.18);     // Orange
        drawSeries('people', 'rgb(96, 165, 250)', 0.22);   // Blue
    }

    function renderTimeLabels(container, duration) {
        /**
         * Generate time labels (e.g. 0:00, 0:30, 1:00...) below the chart.
         */
        container.innerHTML = '';
        const numLabels = Math.min(8, Math.max(3, Math.floor(duration / 15)));
        for (let i = 0; i <= numLabels; i++) {
            const sec = Math.round((i / numLabels) * duration);
            const label = document.createElement('span');
            label.textContent = secondsToTimeString(sec);
            container.appendChild(label);
        }
    }

    // ─── QA Timeline Functions ───
    function timeStringToSeconds(timeStr) {
        /**
         * Convert time string like "0:00:00" or "0:00" to seconds.
         */
        const parts = timeStr.trim().split(':').map(p => parseInt(p, 10));
        if (parts.length === 3) {
            return parts[0] * 3600 + parts[1] * 60 + parts[2];
        } else if (parts.length === 2) {
            return parts[0] * 60 + parts[1];
        }
        return 0;
    }

    function parseTimeRange(evidenceStr) {
        /**
         * Parse "0:00:00 - 0:00:10" format and return {start, end} in seconds.
         */
        const match = evidenceStr.match(/(.+?)\s*-\s*(.+)/);
        if (!match) return null;
        const start = timeStringToSeconds(match[1]);
        const end = timeStringToSeconds(match[2]);
        return { start, end, midpoint: (start + end) / 2 };
    }

    function secondsToTimeString(seconds) {
        /**
         * Convert seconds to "M:SS" or "H:MM:SS" format.
         */
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) {
            return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        }
        return `${m}:${String(s).padStart(2, '0')}`;
    }

    async function loadQATimeline(folder, videoPlayer) {
        /**
         * Fetch QA data from the API and set up the timeline.
         */
        const folderName = folder.split('/').pop();
        try {
            const response = await fetch(`/api/qa-data?folder=${encodeURIComponent(folderName)}`);
            if (!response.ok) {
                console.error('Failed to load QA data');
                return;
            }
            const qaData = await response.json();
            
            if (!qaData || Object.keys(qaData).length === 0) {
                return; // No QA data
            }

            // Set up the QA review section
            const qaReviewSection = document.getElementById('qa-review-section');
            const qaCategoryTabs = document.getElementById('qa-category-tabs');
            const qaCardsList = document.getElementById('qa-cards-list');

            // Clear previous content
            qaCategoryTabs.innerHTML = '';
            qaCardsList.innerHTML = '';

            // Track current QA data using the outer-scope variable
            let currentCategory = null;
            currentQAPairs = [];

            // Create category tabs
            const categories = Object.keys(qaData).sort();
            categories.forEach((category, idx) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'qa-tab';
                if (idx === 0) btn.classList.add('active');
                btn.dataset.category = category;
                btn.textContent = category.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                
                btn.addEventListener('click', () => {
                    // Update active tab
                    qaCategoryTabs.querySelectorAll('.qa-tab').forEach(t => t.classList.remove('active'));
                    btn.classList.add('active');
                    
                    // Render QA pairs for this category
                    currentCategory = category;
                    currentQAPairs = qaData[category] || [];
                    renderQACards(currentQAPairs, videoPlayer);
                    updateTimelineMarkers(currentQAPairs, videoPlayer);
                });
                
                qaCategoryTabs.appendChild(btn);
            });

            // Load the first category by default
            if (categories.length > 0) {
                currentCategory = categories[0];
                currentQAPairs = qaData[currentCategory] || [];
                renderQACards(currentQAPairs, videoPlayer);
                updateTimelineMarkers(currentQAPairs, videoPlayer);
            }

            // Show the QA review section
            qaReviewSection.classList.remove('hidden');

            // Sync video time with QA cards
            videoPlayer.addEventListener('timeupdate', () => {
                syncQACardsWithVideo(currentQAPairs, videoPlayer);
            });

        } catch (error) {
            console.error('Error loading QA timeline:', error);
        }
    }

    function renderQACards(qaPairs, videoPlayer) {
        /**
         * Render QA pairs as interactive cards in the QA cards list.
         */
        const qaCardsList = document.getElementById('qa-cards-list');
        qaCardsList.innerHTML = '';

        if (!qaPairs || qaPairs.length === 0) {
            qaCardsList.innerHTML = '<p style="padding: 1rem; color: var(--text-secondary);">No QA pairs available</p>';
            return;
        }

        qaPairs.forEach((qa, idx) => {
            const timeRange = parseTimeRange(qa['Evidence spans the video']);
            if (!timeRange) return;

            const card = document.createElement('div');
            card.className = 'qa-card';
            card.dataset.startTime = timeRange.start;
            card.dataset.endTime = timeRange.end;

            const question = qa['Question'] || '';
            const answer = qa['Answer'] || '';
            const reasoning = qa['Reasoning type'] || '';
            const difficulty = qa['Difficulty level'] || '';

            card.innerHTML = `
                <div class="qa-card-header">
                    <span class="qa-card-number">Q${idx + 1}</span>
                    <span class="qa-card-timestamp" title="Evidence timespan">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                        ${secondsToTimeString(timeRange.start)} - ${secondsToTimeString(timeRange.end)}
                    </span>
                    <button class="qa-jump-btn" type="button" title="Jump to this moment in the video">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <polyline points="23 4 23 10 17 10"></polyline>
                            <path d="M20.49 15a9 9 0 1 1 .12-13.46L23 10"></path>
                        </svg>
                        Seek
                    </button>
                </div>
                <div class="qa-field">
                    <div class="qa-field-label">Question</div>
                    <div class="qa-question-text">${question}</div>
                </div>
                <div class="qa-field">
                    <div class="qa-field-label">Answer</div>
                    <div class="qa-answer-input" readonly style="border: 1px solid rgba(255,255,255,0.1); background: rgba(15, 23, 42, 0.4); padding: 0.55rem 0.8rem; border-radius: 8px;">${answer}</div>
                </div>
                <div class="qa-field">
                    <div class="qa-field-label">Reasoning</div>
                    <div style="font-size: 0.88rem; color: #a78bfa;">${reasoning}</div>
                </div>
                <div class="qa-field">
                    <div class="qa-field-label">Difficulty</div>
                    <div style="font-size: 0.88rem; color: #a78bfa;">${difficulty}</div>
                </div>
            `;

            card.querySelector('.qa-jump-btn').addEventListener('click', () => {
                videoPlayer.currentTime = timeRange.start;
                videoPlayer.play();
            });

            qaCardsList.appendChild(card);
        });

        syncQACardsWithVideo(qaPairs, videoPlayer);
    }

    function updateTimelineMarkers(qaPairs, videoPlayer) {
        /**
         * Create markers on the timeline bar for each QA pair.
         */
        const timelineTrack = document.getElementById('qa-timeline-track');
        timelineTrack.innerHTML = '';

        if (!videoPlayer.duration || !isFinite(videoPlayer.duration)) return;

        qaPairs.forEach((qa) => {
            const timeRange = parseTimeRange(qa['Evidence spans the video']);
            if (!timeRange) return;

            const percent = (timeRange.midpoint / videoPlayer.duration) * 100;
            const marker = document.createElement('div');
            marker.className = 'qa-timeline-marker';
            marker.style.left = `${percent}%`;
            marker.dataset.category = qa['Reasoning type'] || 'other';
            marker.title = `${qa['Question']?.substring(0, 50)}...`;

            marker.addEventListener('click', () => {
                videoPlayer.currentTime = timeRange.start;
                videoPlayer.play();
            });

            timelineTrack.appendChild(marker);
        });
    }

    function syncQACardsWithVideo(qaPairs, videoPlayer) {
        /**
         * Highlight the QA card that corresponds to the current video time.
         */
        const currentTime = videoPlayer.currentTime;
        const qaCardsList = document.getElementById('qa-cards-list');
        const cards = qaCardsList.querySelectorAll('.qa-card');

        cards.forEach((card) => {
            const startTime = parseFloat(card.dataset.startTime);
            const endTime = parseFloat(card.dataset.endTime);
            
            if (currentTime >= startTime && currentTime <= endTime) {
                if (!card.classList.contains('active')) {
                    card.classList.add('active');
                    // Scroll to the active card
                    const scrollContainer = qaCardsList;
                    const cardRect = card.getBoundingClientRect();
                    const containerRect = scrollContainer.getBoundingClientRect();
                    
                    if (cardRect.bottom > containerRect.bottom) {
                        card.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    } else if (cardRect.top < containerRect.top) {
                        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }
            } else {
                card.classList.remove('active');
            }
        });
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
        
        // Reset Hugging Face elements
        if (hfFilesContainer) hfFilesContainer.classList.add('hidden');
        if (hfFileInput) hfFileInput.innerHTML = '';
        clearHfError();
        
        // Reset tab selection to upload
        if (tabUpload) {
            currentInputTab = 'upload';
            tabUpload.classList.add('active');
            tabHuggingface.classList.remove('active');
            dropZone.classList.remove('hidden');
            hfZone.classList.add('hidden');
        }

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

        // Reset analysis timeline
        document.getElementById('analysis-timeline-section').classList.add('hidden');
        const tlCanvas = document.getElementById('analysis-timeline-canvas');
        if (tlCanvas) {
            const ctx = tlCanvas.getContext('2d');
            ctx.clearRect(0, 0, tlCanvas.width, tlCanvas.height);
        }
        document.getElementById('timeline-playhead').style.display = 'none';
        document.getElementById('timeline-time-labels').innerHTML = '';
        document.getElementById('video-type-badge').classList.add('hidden');
        timelineData = null;
        timelineDuration = 0;
        if (timelineAnimFrame) {
            cancelAnimationFrame(timelineAnimFrame);
            timelineAnimFrame = null;
        }

        // Reset QA review section
        document.getElementById('qa-category-tabs').innerHTML = '';
        document.getElementById('qa-timeline-track').innerHTML = '';
        document.getElementById('qa-cards-list').innerHTML = '';
        document.getElementById('qa-review-section').classList.add('hidden');

        const metaDisplay = document.getElementById('meta-info-display');
        if (metaDisplay) {
            metaDisplay.innerHTML = '';
            metaDisplay.style.display = 'none';
        }

        // Deselect history item
        activeHistoryFolder = null;
        historyList.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));

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
                    showResults(data.results, data.model_info, videoParam, data.analysis_settings, data.object_counts);
                    fileNameDisplay.textContent = videoParam;
                    mainPanel.classList.add('hidden');
                }
            })
            .catch(err => {
                console.log('No preloaded results found:', err.message);
            });
    }

    // Load history on startup
    loadHistory();
});
