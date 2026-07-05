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
    const detectorCustomLocalGroup = document.getElementById('detector_custom_local_group');
    const detectorCustomApiGroup = document.getElementById('detector_custom_api_group');

    const syncModelFields = () => {
        if (!modelTypeSelect) return;
        
        if (modelTypeSelect.value === 'yolo') {
            modelIdSelect.disabled = false;
            modelIdGroup.classList.remove('hidden');
        } else {
            modelIdSelect.disabled = true;
            modelIdGroup.classList.add('hidden');
        }

        if (modelTypeSelect.value === 'custom_local') {
            detectorCustomLocalGroup.classList.remove('hidden');
        } else {
            detectorCustomLocalGroup.classList.add('hidden');
        }

        if (modelTypeSelect.value === 'custom_api') {
            detectorCustomApiGroup.classList.remove('hidden');
        } else {
            detectorCustomApiGroup.classList.add('hidden');
        }
    };
    if (modelTypeSelect) {
        modelTypeSelect.addEventListener('change', syncModelFields);
        syncModelFields(); // Run on startup
    }

    // VLM Model Toggle
    const vlmModelSelect = document.getElementById('vlm_model');
    const vlmGeminiGroup = document.getElementById('vlm_gemini_group');
    const vlmCustomGroup = document.getElementById('vlm_custom_group');

    const syncVlmFields = () => {
        if (!vlmModelSelect) return;
        
        if (vlmModelSelect.value === 'gemini') {
            vlmGeminiGroup.classList.remove('hidden');
            vlmCustomGroup.classList.add('hidden');
        } else if (vlmModelSelect.value === 'custom_vlm') {
            vlmGeminiGroup.classList.add('hidden');
            vlmCustomGroup.classList.remove('hidden');
        } else {
            vlmGeminiGroup.classList.add('hidden');
            vlmCustomGroup.classList.add('hidden');
        }
    };
    if (vlmModelSelect) {
        vlmModelSelect.addEventListener('change', syncVlmFields);
        syncVlmFields();
    }

    // Comparison Dashboard VLM Model Toggle
    const compareVlmSelect = document.getElementById('compare-vlm-model');
    const compareGeminiWrap = document.getElementById('compare-gemini-key-wrap');
    const compareCustomWrap = document.getElementById('compare-custom-vlm-wrap');

    const syncCompareVlmFields = () => {
        if (!compareVlmSelect) return;
        if (compareVlmSelect.value === 'gemini') {
            if (compareGeminiWrap) compareGeminiWrap.classList.remove('hidden');
            if (compareCustomWrap) compareCustomWrap.classList.add('hidden');
        } else if (compareVlmSelect.value === 'custom_vlm') {
            if (compareGeminiWrap) compareGeminiWrap.classList.add('hidden');
            if (compareCustomWrap) compareCustomWrap.classList.remove('hidden');
        } else {
            if (compareGeminiWrap) compareGeminiWrap.classList.add('hidden');
            if (compareCustomWrap) compareCustomWrap.classList.add('hidden');
        }
    };
    if (compareVlmSelect) {
        compareVlmSelect.addEventListener('change', syncCompareVlmFields);
        syncCompareVlmFields();
    }


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
            checkVideoHistoryForCaptions(fileInput.files[0].name);
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

            if (!hfFileInput.dataset.listenerAdded) {
                hfFileInput.addEventListener('change', () => {
                    checkVideoHistoryForCaptions(hfFileInput.value);
                });
                hfFileInput.dataset.listenerAdded = 'true';
            }

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
            if (data.videos.length > 0) {
                checkVideoHistoryForCaptions(hfFileInput.value);
            }
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

    // ─── Auto-generate Captions Disable Logic & Uploader ───
    const autoGenCaptions = document.getElementById('auto_generate_captions');
    const captionsTextarea = document.getElementById('captions');
    if (autoGenCaptions && captionsTextarea) {
        autoGenCaptions.addEventListener('change', () => {
            captionsTextarea.disabled = autoGenCaptions.checked;
            if (autoGenCaptions.checked) {
                captionsTextarea.placeholder = 'Description will be auto-generated by the VLM...';
            } else {
                captionsTextarea.placeholder = 'e.g. A dashcam recording of a vehicle driving down a highway at sunset under rainy weather.';
            }
        });
    }

    const mainUploadCaptionBtn = document.getElementById('main-upload-caption-btn');
    const mainCaptionFileInput = document.getElementById('main-caption-file');
    if (mainUploadCaptionBtn && mainCaptionFileInput) {
        mainUploadCaptionBtn.addEventListener('click', () => {
            mainCaptionFileInput.click();
        });
        mainCaptionFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (event) => {
                let captionText = event.target.result;
                try {
                    const parsed = JSON.parse(captionText);
                    if (parsed.caption) {
                        captionText = parsed.caption;
                    } else if (parsed.captions) {
                        captionText = parsed.captions;
                    } else if (parsed.description) {
                        captionText = parsed.description;
                    } else if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'string') {
                        captionText = parsed[0];
                    }
                } catch (err) {
                    // plain text
                }
                if (captionsTextarea) {
                    captionsTextarea.value = captionText.trim();
                    captionsTextarea.disabled = false;
                }
                if (autoGenCaptions) {
                    autoGenCaptions.checked = false;
                }
            };
            reader.readAsText(file);
        });
    }

    function checkVideoHistoryForCaptions(videoPathOrName) {
        if (!videoPathOrName) return;
        const videoName = videoPathOrName.split('/').pop().split('\\').pop();
        const baseName = videoName.replace(/\.[^/.]+$/, ""); // strip extension
        
        // Find if there is any run for this video baseName in cachedHistoryRuns
        const group = cachedHistoryRuns.find(g => {
            const gName = g.video_name.split('/').pop().split('\\').pop();
            const gBase = gName.replace(/\.[^/.]+$/, "");
            return gBase.toLowerCase() === baseName.toLowerCase();
        });
        
        if (group && group.runs && group.runs.length > 0) {
            let foundCaption = null;
            for (const run of group.runs) {
                if (run.captions && run.captions.trim()) {
                    foundCaption = run.captions.trim();
                    break;
                }
            }
            if (foundCaption) {
                if (captionsTextarea) {
                    captionsTextarea.value = foundCaption;
                    captionsTextarea.disabled = false;
                }
                if (autoGenCaptions) {
                    autoGenCaptions.checked = false;
                }
                return;
            }
        }
        
        // No captions found in history
        if (captionsTextarea) {
            captionsTextarea.value = '';
        }
        if (autoGenCaptions) {
            autoGenCaptions.checked = true;
            if (captionsTextarea) {
                captionsTextarea.disabled = true;
                captionsTextarea.placeholder = 'Description will be auto-generated by the VLM...';
            }
        }
    }

    // Currently active history folder (for highlighting)
    let activeHistoryFolder = null;
    let activeHistoryVideo = null;
    let cachedHistoryRuns = [];

    function loadHistory() {
        fetch('/api/history')
            .then(res => res.json())
            .then(runs => {
                cachedHistoryRuns = runs;
                renderHistoryList(runs);
                if (currentInputTab === 'upload' && fileInput && fileInput.files.length > 0) {
                    checkVideoHistoryForCaptions(fileInput.files[0].name);
                } else if (currentInputTab === 'huggingface' && hfFileInput && hfFileInput.value) {
                    checkVideoHistoryForCaptions(hfFileInput.value);
                }
            })
            .catch(err => {
                console.error('Failed to load history:', err);
            });
    }

    function updateSidebarHighlights() {
        if (!cachedHistoryRuns || cachedHistoryRuns.length === 0) return;
        document.querySelectorAll('.history-group-container').forEach(container => {
            const item = container.querySelector('.history-item');
            const videoName = item.dataset.videoName;
            const group = cachedHistoryRuns.find(g => g.video_name === videoName);
            if (!group) return;
            
            const isGroupActive = group.video_name === activeHistoryVideo || group.runs.some(r => r.folder === activeHistoryFolder);
            if (isGroupActive) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
            
            const dbEl = container.querySelector('.history-nested-dashboard');
            if (dbEl) {
                if (group.video_name === activeHistoryVideo) {
                    dbEl.classList.add('active');
                } else {
                    dbEl.classList.remove('active');
                }
            }
            
            const runEls = container.querySelectorAll('.history-nested-run');
            group.runs.forEach((run, idx) => {
                const runEl = runEls[idx];
                if (runEl) {
                    if (run.folder === activeHistoryFolder) {
                        runEl.classList.add('active');
                    } else {
                        runEl.classList.remove('active');
                    }
                }
            });
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
        runs.forEach(group => {
            const itemContainer = document.createElement('div');
            itemContainer.className = 'history-group-container';
            
            const isGroupActive = group.video_name === activeHistoryVideo || group.runs.some(r => r.folder === activeHistoryFolder);

            const item = document.createElement('div');
            item.className = 'history-item';
            item.style.display = 'flex';
            item.style.alignItems = 'center';
            if (isGroupActive) {
                item.classList.add('active');
            }
            item.dataset.videoName = group.video_name;

            const runCount = group.runs.length;
            const modelBadge = `<span class="meta-badge">${runCount} model${runCount > 1 ? 's' : ''}</span>`;
            
            const chevron = `<svg class="history-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="transition: transform 0.2s; ${isGroupActive ? 'transform: rotate(180deg);' : ''}"><polyline points="6 9 12 15 18 9"></polyline></svg>`;

            item.innerHTML = `
                <div style="flex:1; overflow: hidden;">
                    <div class="history-item-name" title="${group.video_name}">${group.video_name}</div>
                    <div class="history-item-meta">
                        <span>${group.latest_run_date || '—'}</span>
                        ${modelBadge}
                    </div>
                </div>
                ${chevron}
            `;
            
            const nestedList = document.createElement('div');
            nestedList.className = `history-nested-list ${isGroupActive ? '' : 'hidden'}`;
            
            const dashboardDateStr = group.latest_run_date ? ` (${group.latest_run_date})` : '';
            nestedList.innerHTML = `
                <div class="history-nested-dashboard ${group.video_name === activeHistoryVideo ? 'active' : ''}" data-video="${group.video_name}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> Dashboard${dashboardDateStr}
                </div>
            `;
            
            nestedList.querySelector('.history-nested-dashboard').addEventListener('click', (e) => {
                e.stopPropagation();
                activeHistoryVideo = group.video_name;
                activeHistoryFolder = null;
                updateSidebarHighlights();
                loadComparisonDashboard(group.video_name);
            });
            
            group.runs.forEach(run => {
                const runEl = document.createElement('div');
                runEl.className = 'history-nested-run';
                if (run.folder === activeHistoryFolder) {
                    runEl.classList.add('active');
                }
                const dateStr = run.run_date ? ` (${run.run_date})` : '';
                runEl.textContent = (run.model_info ? run.model_info.model_name : run.folder) + dateStr;
                runEl.addEventListener('click', (e) => {
                    e.stopPropagation();
                    activeHistoryVideo = null;
                    activeHistoryFolder = run.folder;
                    updateSidebarHighlights();
                    viewIndividualRun(run);
                });
                nestedList.appendChild(runEl);
            });

            item.addEventListener('click', () => {
                const isHidden = nestedList.classList.contains('hidden');
                historyList.querySelectorAll('.history-nested-list').forEach(l => l.classList.add('hidden'));
                historyList.querySelectorAll('.history-chevron').forEach(c => c.style.transform = 'rotate(0deg)');
                
                if (isHidden) {
                    nestedList.classList.remove('hidden');
                    item.querySelector('.history-chevron').style.transform = 'rotate(180deg)';
                }
            });

            itemContainer.appendChild(item);
            itemContainer.appendChild(nestedList);
            historyList.appendChild(itemContainer);
        });
    }

    let currentComparisonVideo = null;
    let currentComparisonData = null;
    
    // Utility to parse 'HH:MM:SS' or 'MM:SS' to seconds
    function parseTimestampToSeconds(tsStr) {
        if (!tsStr) return 0;
        const parts = tsStr.trim().split(':');
        let secs = 0;
        if (parts.length === 3) {
            secs = parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseFloat(parts[2]);
        } else if (parts.length === 2) {
            secs = parseInt(parts[0]) * 60 + parseFloat(parts[1]);
        }
        return isNaN(secs) ? 0 : secs;
    }

    function drawCompareQATimeline(qaPairs) {
        const timelineTrack = document.getElementById('compare-qa-timeline-track');
        const videoPlayer = document.getElementById('compare-result-video');
        if (!timelineTrack || !videoPlayer) return;
        
        timelineTrack.innerHTML = '';
        if (!videoPlayer.duration || !isFinite(videoPlayer.duration)) {
            videoPlayer.addEventListener('loadedmetadata', () => drawCompareQATimeline(qaPairs), { once: true });
            return;
        }

        qaPairs.forEach((qa) => {
            const timeRangeStr = qa['Evidence spans the video'];
            if (!timeRangeStr) return;

            const times = timeRangeStr.split(' - ');
            if (times.length !== 2) return;
            
            const start = parseTimestampToSeconds(times[0]);
            const end = parseTimestampToSeconds(times[1]);
            const midpoint = (start + end) / 2;

            const percent = (midpoint / videoPlayer.duration) * 100;
            const marker = document.createElement('div');
            marker.className = 'qa-timeline-marker';
            marker.style.left = `${percent}%`;
            marker.dataset.category = qa.Category || qa['Reasoning type'] || 'other';
            marker.title = `${qa.model ? '[' + qa.model + '] ' : ''}${qa['Question']?.substring(0, 50)}...`;

            marker.addEventListener('click', () => {
                videoPlayer.currentTime = start;
                videoPlayer.play();
                
                if (qa.Category) {
                    const tabBtn = document.querySelector(`.qa-compare-tab-btn[data-tab-category="${qa.Category}"]`);
                    if (tabBtn) {
                        tabBtn.click();
                        const layout = document.getElementById('qa-comparison-layout');
                        if (layout) {
                            layout.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    }
                }
            });

            timelineTrack.appendChild(marker);
        });
    }

    // Comparison Dashboard Controls
    async function loadComparisonDashboard(videoName, consensusMethod = 'average') {
        currentComparisonVideo = videoName;
        activeHistoryVideo = videoName;
        activeHistoryFolder = null;
        updateSidebarHighlights();
        
        mainPanel.classList.add('hidden');
        resultsPanel.classList.add('hidden');
        errorPanel.classList.add('hidden');
        document.getElementById('comparison-panel').classList.remove('hidden');
        
        document.getElementById('compare-video-title').textContent = `Comparison Dashboard: ${videoName}`;
        document.getElementById('consensus-method-select').value = consensusMethod;
        
        // Sync VLM selection and credentials from Advanced Settings to Dashboard inputs
        const mainVlmModel = document.getElementById('vlm_model');
        const compareVlmModel = document.getElementById('compare-vlm-model');
        if (mainVlmModel && compareVlmModel) {
            compareVlmModel.value = mainVlmModel.value;
            syncCompareVlmFields();
        }
        const mainGeminiKey = document.getElementById('gemini_api_key');
        const compareGeminiKey = document.getElementById('compare-gemini-key');
        if (mainGeminiKey && compareGeminiKey && mainGeminiKey.value) {
            compareGeminiKey.value = mainGeminiKey.value;
        }
        const mainVlmApiUrl = document.getElementById('vlm_api_url');
        const compareVlmApiUrl = document.getElementById('compare-vlm-api-url');
        if (mainVlmApiUrl && compareVlmApiUrl && mainVlmApiUrl.value) {
            compareVlmApiUrl.value = mainVlmApiUrl.value;
        }
        const mainVlmApiKey = document.getElementById('vlm_api_key');
        const compareVlmApiKey = document.getElementById('compare-vlm-api-key');
        if (mainVlmApiKey && compareVlmApiKey && mainVlmApiKey.value) {
            compareVlmApiKey.value = mainVlmApiKey.value;
        }
        const mainVlmModelId = document.getElementById('vlm_model_id');
        const compareVlmModelId = document.getElementById('compare-vlm-model-id');
        if (mainVlmModelId && compareVlmModelId && mainVlmModelId.value) {
            compareVlmModelId.value = mainVlmModelId.value;
        }

        try {
            const res = await fetch(`/api/video-comparison?video_name=${encodeURIComponent(videoName)}&consensus_method=${consensusMethod}`);
            if (!res.ok) throw new Error('Failed to load comparison data');
            const data = await res.json();
            currentComparisonData = data;
            
            // Populate Video Source Dropdown
            const videoSourceSelect = document.getElementById('compare-video-source');
            if (videoSourceSelect) {
                videoSourceSelect.innerHTML = '';
                data.runs.forEach((run, idx) => {
                    const mName = run.model_info ? run.model_info.model_name : `Run ${idx + 1}`;
                    const selected = idx === 0 ? 'selected' : '';
                    videoSourceSelect.innerHTML += `<option value="${idx}" ${selected}>${mName} Annotated</option>`;
                });
                
                const compareVideo = document.getElementById('compare-result-video');
                const updateCompareVideoSource = () => {
                    const val = videoSourceSelect.value;
                    if (val !== '') {
                        const run = data.runs[parseInt(val)];
                        const path = run.files && run.files.video ? run.files.video : `/output/${run.folder}/result.mp4`;
                        compareVideo.src = streamVideoUrl(path);
                    }
                };
                
                updateCompareVideoSource();
                videoSourceSelect.addEventListener('change', updateCompareVideoSource);
                
                // Sync active items based on time
                compareVideo.addEventListener('timeupdate', () => {
                    if (!compareVideo.duration) return;
                    const currentTime = compareVideo.currentTime;
                    document.querySelectorAll('.qa-compare-item').forEach(item => {
                        const span = item.dataset.timespan;
                        if (!span) return;
                        const times = span.split(' - ');
                        if (times.length === 2) {
                            const start = parseTimestampToSeconds(times[0]);
                            const end = parseTimestampToSeconds(times[1]);
                            if (currentTime >= start && currentTime <= end) {
                                item.style.backgroundColor = 'rgba(139, 92, 246, 0.15)';
                                item.style.borderColor = 'rgba(139, 92, 246, 0.4)';
                            } else {
                                item.style.backgroundColor = '';
                                item.style.borderColor = item.dataset.selected ? 'var(--primary-color)' : 'transparent';
                            }
                        }
                    });
                });
            }
            
            renderComparisonTableAndCharts(data);
            renderQAComparison(data);
            
            // Handle captions
            const captionsArea = document.getElementById('compare-external-captions');
            if (data.verified_data && (data.verified_data.verified_captions || data.verified_data.ground_truth_context)) {
                captionsArea.value = data.verified_data.verified_captions || data.verified_data.ground_truth_context;
            } else {
                const firstWithCaption = data.runs.find(r => r.analysis_settings && r.analysis_settings.captions);
                captionsArea.value = firstWithCaption ? firstWithCaption.analysis_settings.captions : '';
            }
            
        } catch (err) {
            alert('Error loading dashboard: ' + err.message);
        }
    }

    function renderComparisonTableAndCharts(data) {
        const tableHeader = document.getElementById('comparison-table-header');
        const tableBody = document.getElementById('comparison-table-body');
        const chartContainer = document.getElementById('comparison-chart-container');
        
        tableHeader.innerHTML = '<th>Object Class</th>';
        tableBody.innerHTML = '';
        chartContainer.innerHTML = '';
        
        const runs = data.runs;
        runs.forEach(run => {
            const th = document.createElement('th');
            const mName = run.model_info ? run.model_info.model_name : 'Model';
            th.innerHTML = `
                <div class="th-model-wrap">
                    <span>${mName}</span>
                    <button type="button" class="view-player-mini-btn" data-run-idx="${runs.indexOf(run)}">Player</button>
                </div>
            `;
            tableHeader.appendChild(th);
        });
        
        const thConsensus = document.createElement('th');
        thConsensus.textContent = 'Consensus';
        tableHeader.appendChild(thConsensus);
        
        const thVerified = document.createElement('th');
        thVerified.textContent = 'Ground Truth';
        tableHeader.appendChild(thVerified);
        
        // Listen to player button clicks
        setTimeout(() => {
            document.querySelectorAll('.view-player-mini-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const idx = parseInt(btn.dataset.runIdx, 10);
                    const run = runs[idx];
                    viewIndividualRun(run);
                });
            });
        }, 50);
        
        const classes = new Set();
        runs.forEach(run => {
            Object.keys(run.object_counts || {}).forEach(cls => classes.add(cls));
        });
        
        const sortedClasses = Array.from(classes).sort();
        
        if (sortedClasses.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-secondary);">No objects detected by any model.</td></tr>';
            return;
        }
        
        sortedClasses.forEach(cls => {
            const tr = document.createElement('tr');
            
            const tdClass = document.createElement('td');
            tdClass.innerHTML = `<strong>${cls}</strong>`;
            tr.appendChild(tdClass);
            
            let maxCount = 1;
            runs.forEach(run => {
                const count = (run.object_counts && run.object_counts[cls]) || 0;
                maxCount = Math.max(maxCount, count);
                const td = document.createElement('td');
                td.textContent = count;
                tr.appendChild(td);
            });
            
            const consensus = data.consensus_counts[cls] || 0;
            maxCount = Math.max(maxCount, consensus);
            const tdConsensus = document.createElement('td');
            tdConsensus.className = 'consensus-count-val';
            tdConsensus.textContent = consensus;
            tr.appendChild(tdConsensus);
            
            const verifiedVal = (data.verified_data && 
                ((data.verified_data.verified_counts && data.verified_data.verified_counts[cls] !== undefined) ||
                 (data.verified_data.ground_truth_counts && data.verified_data.ground_truth_counts[cls] !== undefined)))
                ? (data.verified_data.ground_truth_counts || data.verified_data.verified_counts)[cls]
                : consensus;
            const tdVerified = document.createElement('td');
            tdVerified.innerHTML = `<input type="number" min="0" class="verified-count-input" data-class="${cls}" value="${verifiedVal}" style="width: 80px; text-align: center; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15); background: rgba(15,23,42,0.6); color: #fff; padding: 0.3rem;">`;
            tr.appendChild(tdVerified);
            
            tableBody.appendChild(tr);
            
            const chartRow = document.createElement('div');
            chartRow.className = 'chart-row';
            
            let barsHtml = '';
            runs.forEach((run, idx) => {
                const count = (run.object_counts && run.object_counts[cls]) || 0;
                const pct = (count / maxCount) * 100;
                const mName = run.model_info ? run.model_info.model_name : 'Model';
                barsHtml += `
                    <div class="chart-bar-wrap">
                        <span class="bar-legend-name">${mName}</span>
                        <div class="chart-bar-fill model-color-${idx % 5}" style="width: ${pct}%; min-width: 8px;">${count}</div>
                    </div>
                `;
            });
            
            const consensusPct = (consensus / maxCount) * 100;
            barsHtml += `
                <div class="chart-bar-wrap consensus-bar-wrap">
                    <span class="bar-legend-name">Consensus</span>
                    <div class="chart-bar-fill consensus-bar" style="width: ${consensusPct}%; min-width: 8px;">${consensus}</div>
                </div>
            `;
            
            chartRow.innerHTML = `
                <div class="chart-row-header">
                    <span class="chart-row-title">${cls}</span>
                </div>
                <div class="chart-row-bars">
                    ${barsHtml}
                </div>
            `;
            chartContainer.appendChild(chartRow);
        });
    }

    function viewIndividualRun(run) {
        document.getElementById('comparison-panel').classList.add('hidden');
        activeHistoryVideo = null;
        activeHistoryFolder = run.folder;
        updateSidebarHighlights();
        
        const normalizedResults = {
            folder: `/output/${run.folder}`,
            video: run.files.video,
            is_original_video: run.files.is_original_video || false,
            csv: run.files.csv,
            json: run.files.json,
            qa_json_files: run.files.qa_json_files || [],
        };
        
        showResults(normalizedResults, run.model_info, run.video_name || run.folder, run.analysis_settings, run.object_counts);
    }

    async function renderQAComparison(data) {
        const modelContainer = document.getElementById('model-qa-compare-container');
        const verifiedContainer = document.getElementById('verified-qa-container');
        if (modelContainer) modelContainer.innerHTML = '';
        if (verifiedContainer) verifiedContainer.innerHTML = '';
        
        const runs = data.runs;
        const runsContainer = document.createElement('div');
        runsContainer.className = 'qa-models-comparison';
        

        runsContainer.innerHTML = '';
        
        const tabHeader = document.createElement('div');
        tabHeader.className = 'qa-compare-tab-header';
        
        const tabContent = document.createElement('div');
        tabContent.className = 'qa-compare-tab-content';
        
        // Fetch QA data for all runs
        const qaResults = await Promise.all(runs.map(run => 
            fetch(`/api/qa-data?folder=${encodeURIComponent(run.folder)}`).then(res => res.json()).catch(() => ({}))
        ));
        


        // Gather all unique categories
        const categories = new Set();
        qaResults.forEach(qaData => {
            Object.keys(qaData).forEach(cat => {
                if (qaData[cat] && qaData[cat].length > 0) categories.add(cat);
            });
        });

        if (categories.size === 0) {
            tabContent.innerHTML = '<p style="color:var(--text-secondary);padding:1rem;">No QA pairs generated by any model.</p>';
        } else {
            let firstTab = true;
            Array.from(categories).sort().forEach(cat => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = `qa-compare-tab-btn ${firstTab ? 'active' : ''}`;
                btn.dataset.tabCategory = cat;
                btn.textContent = cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                
                const catPanel = document.createElement('div');
                catPanel.className = `qa-compare-run-list ${firstTab ? '' : 'hidden'}`;
                
                const columnsContainer = document.createElement('div');
                columnsContainer.style.display = 'flex';
                columnsContainer.style.gap = '1rem';
                columnsContainer.style.overflowX = 'auto';
                columnsContainer.style.paddingBottom = '1rem';

                runs.forEach((run, idx) => {
                    const mName = run.model_info ? run.model_info.model_name : `Run ${idx + 1}`;
                    const col = document.createElement('div');
                    col.style.flex = '1';
                    col.style.minWidth = '300px';
                    col.style.background = 'rgba(255, 255, 255, 0.02)';
                    col.style.padding = '1rem';
                    col.style.borderRadius = '8px';
                    col.style.border = '1px solid rgba(255, 255, 255, 0.05)';
                    
                    const colTitle = document.createElement('h4');
                    colTitle.textContent = mName;
                    colTitle.style.marginTop = '0';
                    colTitle.style.marginBottom = '1rem';
                    colTitle.style.color = 'var(--text-primary)';
                    colTitle.style.fontSize = '0.9rem';
                    col.appendChild(colTitle);

                    const qasList = qaResults[idx][cat] || [];
                    if (qasList.length === 0) {
                        col.innerHTML += '<p style="color:var(--text-secondary); font-size:0.8rem;">No questions generated.</p>';
                    } else {
                        qasList.forEach(qa => {
                            const item = document.createElement('div');
                            item.className = 'qa-compare-item';
                            item.style.cursor = 'pointer';
                            item.style.border = '1px solid transparent';
                            item.style.transition = 'all 0.2s';
                            item.innerHTML = `
                                <div class="qa-compare-q" style="margin-bottom: 0.5rem;"><strong>Q:</strong> ${qa.Question}</div>
                                <div class="qa-compare-a"><strong>A:</strong> ${qa.Answer} <span class="qa-compare-meta">(${qa['Evidence spans the video']})</span></div>
                            `;
                            item.dataset.timespan = qa['Evidence spans the video'];
                            
                            item.addEventListener('mouseover', () => {
                                if(!item.dataset.selected) {
                                    item.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                                }
                            });
                            item.addEventListener('mouseout', () => {
                                if(!item.dataset.selected) {
                                    item.style.borderColor = 'transparent';
                                }
                            });
                            
                            item.addEventListener('click', () => {
                                const span = qa['Evidence spans the video'];
                                if (span) {
                                    const times = span.split(' - ');
                                    if (times.length === 2) {
                                        const start = parseTimestampToSeconds(times[0]);
                                        const video = document.getElementById('compare-result-video');
                                        if (video) {
                                            video.currentTime = start;
                                            video.play();
                                        }
                                    }
                                }
                                document.querySelectorAll('.qa-compare-item').forEach(el => {
                                    el.dataset.selected = '';
                                    el.style.borderColor = 'transparent';
                                });
                                item.dataset.selected = 'true';
                                item.style.borderColor = 'var(--primary-color)';
                            });
                            
                            col.appendChild(item);
                        });
                    }
                    columnsContainer.appendChild(col);
                });
                
                catPanel.appendChild(columnsContainer);

                btn.addEventListener('click', () => {
                    runsContainer.querySelectorAll('.qa-compare-tab-btn').forEach(b => b.classList.remove('active'));
                    runsContainer.querySelectorAll('.qa-compare-run-list').forEach(l => l.classList.add('hidden'));
                    btn.classList.add('active');
                    catPanel.classList.remove('hidden');

                    const currentTimelineData = [];
                    runs.forEach((run, idx) => {
                        const mName = run.model_info ? run.model_info.model_name : `Run ${idx + 1}`;
                        const qasList = qaResults[idx][cat] || [];
                        qasList.forEach(qa => {
                            if (qa['Evidence spans the video']) {
                                currentTimelineData.push({
                                    'Evidence spans the video': qa['Evidence spans the video'],
                                    'Reasoning type': qa['Reasoning type'] || 'other',
                                    'Category': cat,
                                    'Question': qa.Question,
                                    model: mName
                                });
                            }
                        });
                    });
                    drawCompareQATimeline(currentTimelineData);
                });
                
                if (firstTab) {
                    btn.classList.add('active');
                    catPanel.classList.remove('hidden');
                    
                    const currentTimelineData = [];
                    runs.forEach((run, idx) => {
                        const mName = run.model_info ? run.model_info.model_name : `Run ${idx + 1}`;
                        const qasList = qaResults[idx][cat] || [];
                        qasList.forEach(qa => {
                            if (qa['Evidence spans the video']) {
                                currentTimelineData.push({
                                    'Evidence spans the video': qa['Evidence spans the video'],
                                    'Reasoning type': qa['Reasoning type'] || 'other',
                                    'Category': cat,
                                    'Question': qa.Question,
                                    model: mName
                                });
                            }
                        });
                    });
                    setTimeout(() => drawCompareQATimeline(currentTimelineData), 500);
                    firstTab = false;
                }
                
                tabHeader.appendChild(btn);
                tabContent.appendChild(catPanel);
            });
        }
        runsContainer.appendChild(tabHeader);
        runsContainer.appendChild(tabContent);

        if (modelContainer) modelContainer.appendChild(runsContainer);
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
        const isAutoGenCaptions = document.getElementById('auto_generate_captions') && document.getElementById('auto_generate_captions').checked;

        // Set auto_generate_captions field in form payload
        formData.set('auto_generate_captions', isAutoGenCaptions ? 'true' : 'false');

        const activeQaCategories = [];
        if (document.getElementById('qa_counting').checked) activeQaCategories.push('counting');
        // qa_negative, qa_ambiguity, qa_day_night are disabled — only counting is active

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
                } else if (data.status === 'generating_captions') {
                    loadingStatus.textContent = 'Analyzing video context with Qwen2-VL...';
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
        if (results.folder) {
            const folderName = results.folder.replace(/^\/output\//, '').split('/').pop();
            activeHistoryFolder = folderName;
            activeHistoryVideo = null;
            updateSidebarHighlights();
        }

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
        
        // Reset Captions text area
        const captionsTextarea = document.getElementById('captions');
        if (captionsTextarea) {
            captionsTextarea.disabled = false;
            captionsTextarea.placeholder = 'e.g. A dashcam recording of a vehicle driving down a highway at sunset under rainy weather.';
        }
        
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
        document.getElementById('comparison-panel').classList.add('hidden');

        const metaDisplay = document.getElementById('meta-info-display');
        if (metaDisplay) {
            metaDisplay.innerHTML = '';
            metaDisplay.style.display = 'none';
        }

        // Deselect history item
        activeHistoryVideo = null;
        activeHistoryFolder = null;
        updateSidebarHighlights();
        
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

    // Comparison Dashboard Controls
    const consensusSelect = document.getElementById('consensus-method-select');
    if (consensusSelect) {
        consensusSelect.addEventListener('change', (e) => {
            if (currentComparisonVideo) {
                loadComparisonDashboard(currentComparisonVideo, e.target.value);
            }
        });
    }

    const compareCaptionFile = document.getElementById('compare-caption-file');
    const compareUploadCaptionBtn = document.getElementById('compare-upload-caption-btn');
    const compareExternalCaptions = document.getElementById('compare-external-captions');

    if (compareUploadCaptionBtn && compareCaptionFile && compareExternalCaptions) {
        compareUploadCaptionBtn.addEventListener('click', () => {
            compareCaptionFile.click();
        });

        compareCaptionFile.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (evt) => {
                const content = evt.target.result;
                if (file.name.endsWith('.json')) {
                    try {
                        const parsed = JSON.parse(content);
                        if (typeof parsed === 'string') {
                            compareExternalCaptions.value = parsed;
                        } else if (Array.isArray(parsed)) {
                            compareExternalCaptions.value = parsed.join('\n');
                        } else if (parsed && typeof parsed === 'object') {
                            const caption = parsed.caption || parsed.captions || parsed.description || parsed.text || JSON.stringify(parsed, null, 2);
                            compareExternalCaptions.value = typeof caption === 'object' ? JSON.stringify(caption, null, 2) : caption;
                        } else {
                            compareExternalCaptions.value = content;
                        }
                    } catch (err) {
                        compareExternalCaptions.value = content;
                    }
                } else {
                    compareExternalCaptions.value = content;
                }
            };
            reader.readAsText(file);
            compareCaptionFile.value = '';
        });
    }


    const runAllModelsBtn = document.getElementById('run-all-models-btn');
    if (runAllModelsBtn) {
        runAllModelsBtn.addEventListener('click', () => {
            alert("To run all models, please go to 'New Analysis', upload your video and select 'Run All Available Models' from the Model Type dropdown.");
            resetApp();
        });
    }

    const compareCloseBtn = document.getElementById('compare-close-btn');
    if (compareCloseBtn) {
        compareCloseBtn.addEventListener('click', () => {
            resetApp();
        });
    }

    const saveVerifiedBtn = document.getElementById('save-verified-btn');
    if (saveVerifiedBtn) {
        saveVerifiedBtn.addEventListener('click', async () => {
            if (!currentComparisonVideo) return;
            
            // Gather ground truth counts from input fields
            const ground_truth_counts = {};
            document.querySelectorAll('.verified-count-input').forEach(input => {
                const cls = input.dataset.class;
                const val = parseInt(input.value, 10);
                ground_truth_counts[cls] = isNaN(val) ? 0 : val;
            });
            
            const ground_truth_context = document.getElementById('compare-external-captions').value.trim();
            
            try {
                const res = await fetch('/api/save-verified', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        video_name: currentComparisonVideo,
                        ground_truth_counts,
                        ground_truth_context
                    })
                });
                
                if (!res.ok) throw new Error('Failed to save ground truth data');
                alert('Ground truth saved successfully!');
                loadHistory();
            } catch (err) {
                alert('Error saving: ' + err.message);
            }
        });
    }

    // compareRegenQaBtn removed — QA regeneration via LLM is no longer available
    // The QA system is now purely rule-based (counting only)

    // Load history on startup
    loadHistory();
});
