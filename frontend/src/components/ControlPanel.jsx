import React, { useState } from 'react';
import { 
  Sliders, 
  Filter, 
  Camera, 
  Bell, 
  Save, 
  Trash2, 
  Download, 
  ChevronDown, 
  ChevronUp, 
  Mail, 
  Settings,
  Lock
} from 'lucide-react';

function ControlPanel({ 
  settings, 
  updateSetting, 
  applyPreset, 
  clearHistory, 
  userRole,
  materials,
  selectedMaterial,
  onSelectMaterial,
  onSavePreset
}) {
  const [openSection, setOpenSection] = useState('parameters'); // parameters, filters, camera, alerts, data
  const isReadOnly = userRole === 'operator';

  const toggleSection = (sectionName) => {
    setOpenSection(openSection === sectionName ? null : sectionName);
  };

  // Export Data Handler (CSV / JSON)
  const exportData = () => {
    const dataFormat = settings.dataManagement.exportFormat;
    const dateStart = settings.dataManagement.dateRangeStart;
    const dateEnd = settings.dataManagement.dateRangeEnd;

    // Filter defect history by date range
    const filteredHistory = defectHistory.filter(d => {
      const dDate = new Date(d.timestamp).toISOString().split('T')[0];
      return dDate >= dateStart && dDate <= dateEnd;
    });

    if (filteredHistory.length === 0) {
      alert("No defect records found in the specified date range.");
      return;
    }

    let fileContent = '';
    let mimeType = '';
    let fileName = `defect_report_${dateStart}_to_${dateEnd}`;

    if (dataFormat === 'json') {
      fileContent = JSON.stringify(filteredHistory, null, 2);
      mimeType = 'application/json';
      fileName += '.json';
    } else {
      // CSV
      const headers = ['ID', 'Type', 'Confidence', 'Camera', 'Size (mm)', 'Location', 'Timestamp'];
      const rows = filteredHistory.map(d => [
        d.id,
        d.type,
        d.confidence,
        d.camera,
        d.size_mm,
        d.location,
        new Date(d.timestamp).toISOString()
      ]);
      fileContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
      mimeType = 'text/csv';
      fileName += '.csv';
    }

    const blob = new Blob([fileContent], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-4 h-full text-white">
      
      {/* Read-only restriction banner */}
      {isReadOnly && (
        <div className="bg-orange/15 border border-orange/35 p-3.5 rounded-2xl flex items-start gap-3 text-orange">
          <Lock className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="flex flex-col">
            <span className="text-[10px] font-bold uppercase tracking-wider">Access Restricted (Read-Only)</span>
            <span className="text-[9px] text-slate-300 font-sans mt-0.5">
              Logged in as Operator. Please authenticate as a Manager or Administrator in the top right to modify QC configurations.
            </span>
          </div>
        </div>
      )}

      {/* Kizen Inspection Engine Status Card */}
      <div className="glass-panel p-4 rounded-2xl border border-white/5 shadow-md flex flex-col gap-3">
        <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <Settings className="w-3.5 h-3.5 text-cyan" />
          <span>Kizen Vision Engine</span>
        </h3>
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] text-gray uppercase font-semibold">Hardware Link</span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              <span className="text-[10px] font-bold text-green font-mono">
                ONLINE • GENUINE
              </span>
            </div>
          </div>
          <a
            href="https://kizen.co.in/"
            target="_blank"
            rel="noopener noreferrer"
            className="py-1.5 px-3 rounded-xl text-[9px] font-extrabold tracking-wider uppercase border border-cyan/40 bg-cyan/10 text-cyan hover:bg-cyan/20 transition-all no-underline"
          >
            kizen.co.in
          </a>
        </div>
      </div>

      {/* Presets Selection Card */}
      <div className="glass-panel p-4 rounded-2xl border border-white/5 shadow-md flex flex-col gap-3">
        <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <Save className="w-3.5 h-3.5 text-cyan" />
          <span>Material Presets & Settings</span>
        </h3>
        
        <div className="flex flex-col gap-2">
          <div className="flex flex-col gap-1">
            <label className="text-[8px] font-bold text-gray uppercase">Active Material Presets</label>
            <select
              value={selectedMaterial}
              onChange={(e) => onSelectMaterial(e.target.value)}
              className="bg-navy border border-white/10 rounded-xl px-3 py-2 text-xs text-white font-medium focus:outline-none cursor-pointer w-full"
            >
              {materials.map(m => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </div>
          
          {!isReadOnly && (
            <button
              onClick={onSavePreset}
              className="mt-1 w-full py-2 bg-blue-600 hover:bg-blue-700 text-white font-extrabold text-[10px] tracking-wider uppercase rounded-xl flex items-center justify-center gap-1.5 shadow-sm shadow-blue-600/20 transition-all cursor-pointer hover:scale-103 active:scale-97"
            >
              <Save className="w-3.5 h-3.5" />
              <span>Save Changes to Preset</span>
            </button>
          )}
        </div>
      </div>

      {/* Accordion Configurations */}
      <div className="flex-1 flex flex-col gap-2">
        
        {/* Section 1: Detection Parameters */}
        <div className="glass-panel rounded-2xl border border-white/5 overflow-hidden">
          <button 
            onClick={() => toggleSection('parameters')}
            className="w-full px-4 py-3 flex justify-between items-center text-[10px] font-bold tracking-widest uppercase bg-white/[0.01] hover:bg-white/[0.03] text-white"
          >
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-cyan" />
              <span>Inspection Parameters</span>
            </div>
            {openSection === 'parameters' ? <ChevronUp className="w-4 h-4 text-gray" /> : <ChevronDown className="w-4 h-4 text-gray" />}
          </button>

          {openSection === 'parameters' && (
            <div className="p-4 border-t border-white/5 flex flex-col gap-4 bg-navy/20 font-mono text-[10px]">
              
              {/* Confidence Threshold */}
              <div>
                <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
                  <span>Confidence Limit</span>
                  <span className="text-white font-mono font-bold">{settings.confidenceThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0.10" max="0.90" step="0.05"
                  value={settings.confidenceThreshold}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('confidenceThreshold', parseFloat(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                />
              </div>

              {/* IoU Threshold */}
              <div>
                <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
                  <span>IoU Threshold (NMS)</span>
                  <span className="text-white font-mono font-bold">{settings.iouThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0.10" max="0.90" step="0.05"
                  value={settings.iouThreshold}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('iouThreshold', parseFloat(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                />
              </div>

              {/* Min Defect Area */}
              <div>
                <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
                  <span>Min Defect Size</span>
                  <span className="text-white font-mono font-bold">{settings.minDefectArea.toFixed(1)} mm</span>
                </div>
                <input
                  type="range" min="0.5" max="10.0" step="0.5"
                  value={settings.minDefectArea}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('minDefectArea', parseFloat(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                />
              </div>

              {/* Max Defect Area */}
              <div>
                <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
                  <span>Max Defect Size</span>
                  <span className="text-white font-mono font-bold">{settings.maxDefectArea.toFixed(0)} mm</span>
                </div>
                <input
                  type="range" min="10.0" max="1000.0" step="10.0"
                  value={settings.maxDefectArea}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('maxDefectArea', parseFloat(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                />
              </div>

              {/* Fabric Cuttable Width (ASTM D5430) */}
              <div>
                <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
                  <span>Cuttable Width</span>
                  <span className="text-white font-mono font-bold">{settings.fabricWidth || 1800} mm</span>
                </div>
                <input
                  type="range" min="1000" max="3000" step="100"
                  value={settings.fabricWidth || 1800}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('fabricWidth', parseInt(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                />
              </div>

              {/* ASTM Points Tolerance Limit */}
              <div>
                <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
                  <span>ASTM Tolerance Limit</span>
                  <span className="text-white font-mono font-bold">{settings.maxAllowedPoints || 40} pts</span>
                </div>
                <input
                  type="range" min="10" max="100" step="5"
                  value={settings.maxAllowedPoints || 40}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('maxAllowedPoints', parseInt(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                />
              </div>
            </div>
          )}
        </div>

        {/* Section 2: Defect Type Filters */}
        <div className="glass-panel rounded-2xl border border-white/5 overflow-hidden">
          <button 
            onClick={() => toggleSection('filters')}
            className="w-full px-4 py-3 flex justify-between items-center text-[10px] font-bold tracking-widest uppercase bg-white/[0.01] hover:bg-white/[0.03] text-white"
          >
            <div className="flex items-center gap-2">
              <Filter className="w-3.5 h-3.5 text-cyan" />
              <span>Category Filters</span>
            </div>
            {openSection === 'filters' ? <ChevronUp className="w-4 h-4 text-gray" /> : <ChevronDown className="w-4 h-4 text-gray" />}
          </button>

          {openSection === 'filters' && (
            <div className="p-4 border-t border-white/5 flex flex-col gap-3 bg-navy/20">
              {Object.keys(settings.defectFilters).map((type) => {
                const filter = settings.defectFilters[type];
                return (
                  <div key={type} className="p-2.5 rounded-xl border border-white/5 bg-navy/35 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-white font-mono">{type}</span>
                      
                      <button
                        onClick={() => !isReadOnly && updateSetting(`defectFilters.${type}.enabled`, !filter.enabled)}
                        disabled={isReadOnly}
                        className="cursor-pointer focus:outline-none disabled:opacity-50"
                      >
                        <div className={`w-8 h-4 rounded-full p-0.5 transition-all duration-150 ${filter.enabled ? 'bg-cyan' : 'bg-navy/80'}`}>
                          <div className={`w-3 h-3 rounded-full bg-white transition-all duration-150 ${filter.enabled ? 'translate-x-4 shadow-sm' : 'translate-x-0'}`}></div>
                        </div>
                      </button>
                    </div>

                    {filter.enabled && (
                      <div className="grid grid-cols-2 gap-2 mt-1 border-t border-white/5 pt-2">
                        {/* Conf slider */}
                        <div className="flex flex-col">
                          <span className="text-[8px] font-bold text-gray uppercase font-sans">Min Confidence</span>
                          <div className="flex items-center gap-1.5 mt-0.5">
                            <input
                              type="range" min="0.10" max="0.90" step="0.05"
                              value={filter.threshold}
                              disabled={isReadOnly}
                              onChange={(e) => updateSetting(`defectFilters.${type}.threshold`, parseFloat(e.target.value))}
                              className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                            />
                            <span className="text-[9px] font-mono font-bold text-white">{filter.threshold.toFixed(2)}</span>
                          </div>
                        </div>
                        {/* Priority / Alert toggle */}
                        <div className="flex justify-between items-center text-[9px] border border-white/5 rounded-lg px-2 py-0.5 bg-navy/20">
                          <span className="text-gray uppercase font-sans font-semibold">Alarm alert:</span>
                          <input
                            type="checkbox"
                            checked={filter.alert}
                            disabled={isReadOnly}
                            onChange={(e) => updateSetting(`defectFilters.${type}.alert`, e.target.checked)}
                            className="w-3 h-3 accent-cyan cursor-pointer disabled:opacity-50"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Section 3: Camera Controls */}
        <div className="glass-panel rounded-2xl border border-white/5 overflow-hidden">
          <button 
            onClick={() => toggleSection('camera')}
            className="w-full px-4 py-3 flex justify-between items-center text-[10px] font-bold tracking-widest uppercase bg-white/[0.01] hover:bg-white/[0.03] text-white"
          >
            <div className="flex items-center gap-2">
              <Camera className="w-3.5 h-3.5 text-cyan" />
              <span>Line-Scan Hardware Capture</span>
            </div>
            {openSection === 'camera' ? <ChevronUp className="w-4 h-4 text-gray" /> : <ChevronDown className="w-4 h-4 text-gray" />}
          </button>

          {openSection === 'camera' && (
            <div className="p-4 border-t border-white/5 flex flex-col gap-3 bg-navy/20">
              
              {/* Hardware Sensor Identification */}
              <div className="flex flex-col gap-1 p-2.5 rounded-xl border border-cyan/20 bg-cyan/5">
                <div className="flex items-center justify-between">
                  <label className="text-[8px] font-bold text-cyan uppercase font-sans tracking-wider">Line-Scan Interface</label>
                  <span className="text-[9px] font-mono font-bold text-emerald">169.254.231.206</span>
                </div>
                <div className="text-xs font-semibold text-white">
                  ChinaVision GELM44M-T2 (GigE Line-Scan)
                </div>
                <div className="text-[9px] text-slate-400 font-mono">
                  1000+ Hz Line Rate · 4096 px Sensor Width
                </div>
              </div>

              {['cam1'].map((camId) => {
                const cam = settings.cameraControls[camId];
                return (
                  <div key={camId} className="p-2.5 rounded-xl border border-white/5 bg-navy/35 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-white font-mono">
                        SENSOR CONTROLS
                      </span>
                      
                      <button
                        onClick={() => !isReadOnly && updateSetting(`cameraControls.${camId}.enabled`, !cam.enabled)}
                        disabled={isReadOnly}
                        className="cursor-pointer focus:outline-none disabled:opacity-50"
                      >
                        <div className={`w-8 h-4 rounded-full p-0.5 transition-all duration-150 ${cam.enabled ? 'bg-cyan' : 'bg-navy/80'}`}>
                          <div className={`w-3 h-3 rounded-full bg-white transition-all duration-150 ${cam.enabled ? 'translate-x-4 shadow-sm' : 'translate-x-0'}`}></div>
                        </div>
                      </button>
                    </div>

                    {cam.enabled && (
                      <div className="flex flex-col gap-2 mt-1 border-t border-white/5 pt-2 font-mono">
                        {/* Analog Gain */}
                        <div>
                          <div className="flex justify-between text-[8px] font-semibold text-gray uppercase mb-0.5 font-sans">
                            <span>Analog Gain</span>
                            <span className="text-white font-mono font-bold">{cam.gain || 16}x</span>
                          </div>
                          <input
                            type="range" min="8" max="64" step="2"
                            value={cam.gain || 16}
                            disabled={isReadOnly}
                            onChange={(e) => updateSetting(`cameraControls.${camId}.gain`, parseInt(e.target.value))}
                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                          />
                        </div>

                        {/* Exposure Time */}
                        <div>
                          <div className="flex justify-between text-[8px] font-semibold text-gray uppercase mb-0.5 font-sans">
                            <span>Exposure Time</span>
                            <span className="text-white font-mono font-bold">{cam.exposure || 15} ms</span>
                          </div>
                          <input
                            type="range" min="1" max="100" step="1"
                            value={cam.exposure || 15}
                            disabled={isReadOnly}
                            onChange={(e) => updateSetting(`cameraControls.${camId}.exposure`, parseInt(e.target.value))}
                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                          />
                        </div>

                        {/* Slice Height */}
                        <div>
                          <div className="flex justify-between text-[8px] font-semibold text-gray uppercase mb-0.5 font-sans">
                            <span>Slice Height (ROI)</span>
                            <span className="text-white font-mono font-bold">{cam.slice_height || 256} px</span>
                          </div>
                          <input
                            type="range" min="64" max="512" step="32"
                            value={cam.slice_height || 256}
                            disabled={isReadOnly}
                            onChange={(e) => updateSetting(`cameraControls.${camId}.slice_height`, parseInt(e.target.value))}
                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                          />
                        </div>

                        {/* Image enhance */}
                        <div className="grid grid-cols-1 gap-2 mt-1">
                          <button
                            onClick={() => !isReadOnly && updateSetting(`cameraControls.${camId}.enhance`, !cam.enhance)}
                            disabled={isReadOnly}
                            className={`py-1.5 rounded-xl text-[8px] font-bold uppercase tracking-wider border cursor-pointer disabled:opacity-50 ${cam.enhance ? 'bg-cyan/15 border-cyan text-cyan' : 'bg-navy/40 border-white/5 text-gray hover:text-white'}`}
                          >
                            GPU / CPU CLAHE Enhancement
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Section 4: Alert Configurations */}
        <div className="glass-panel rounded-2xl border border-white/5 overflow-hidden">
          <button 
            onClick={() => toggleSection('alerts')}
            className="w-full px-4 py-3 flex justify-between items-center text-[10px] font-bold tracking-widest uppercase bg-white/[0.01] hover:bg-white/[0.03] text-white"
          >
            <div className="flex items-center gap-2">
              <Bell className="w-3.5 h-3.5 text-cyan" />
              <span>Alarm Configuration</span>
            </div>
            {openSection === 'alerts' ? <ChevronUp className="w-4 h-4 text-gray" /> : <ChevronDown className="w-4 h-4 text-gray" />}
          </button>

          {openSection === 'alerts' && (
            <div className="p-4 border-t border-white/5 flex flex-col gap-3.5 bg-navy/20 font-mono text-[10px]">
              {/* Sound notification check */}
              <div className="flex items-center justify-between text-[10px] font-semibold text-gray uppercase font-sans">
                <span>Sound Buzzer</span>
                <input
                  type="checkbox"
                  checked={settings.alerts.soundEnabled}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('alerts.soundEnabled', e.target.checked)}
                  className="w-3.5 h-3.5 accent-cyan cursor-pointer disabled:opacity-50"
                />
              </div>

              {/* Visual Flash overlay check */}
              <div className="flex items-center justify-between text-[10px] font-semibold text-gray uppercase font-sans">
                <span>Visual Flash Strobe</span>
                <input
                  type="checkbox"
                  checked={settings.alerts.flashEnabled}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('alerts.flashEnabled', e.target.checked)}
                  className="w-3.5 h-3.5 accent-cyan cursor-pointer disabled:opacity-50"
                />
              </div>

              {/* Cooldown slider */}
              <div>
                <div className="flex justify-between text-[9px] font-semibold text-gray uppercase mb-1 font-sans">
                  <span>Alarm Cooldown</span>
                  <span className="text-white font-mono font-bold">{settings.alerts.cooldown}s</span>
                </div>
                <input
                  type="range" min="1" max="30" step="1"
                  value={settings.alerts.cooldown}
                  disabled={isReadOnly}
                  onChange={(e) => updateSetting('alerts.cooldown', parseInt(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan disabled:opacity-50"
                />
              </div>

              {/* Email alerting */}
              <div className="flex flex-col gap-1 border-t border-white/5 pt-2">
                <span className="text-[8px] font-bold text-gray uppercase font-sans mb-1">Alert Recipients</span>
                <div className="relative flex items-center">
                  <Mail className="absolute left-2.5 w-3.5 h-3.5 text-gray" />
                  <input
                    type="email"
                    value={settings.alerts.email}
                    disabled={isReadOnly}
                    onChange={(e) => updateSetting('alerts.email', e.target.value)}
                    placeholder="email@example.com"
                    className="w-full pl-8 pr-2 py-1.5 bg-navy/60 border border-white/10 rounded-lg text-[10px] font-medium text-white focus:outline-none focus:border-cyan placeholder:text-gray/40 disabled:opacity-50"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Section 5: Data Management */}
        <div className="glass-panel rounded-2xl border border-white/5 overflow-hidden">
          <button 
            onClick={() => toggleSection('data')}
            className="w-full px-4 py-3 flex justify-between items-center text-[10px] font-bold tracking-widest uppercase bg-white/[0.01] hover:bg-white/[0.03] text-white"
          >
            <div className="flex items-center gap-2">
              <Settings className="w-3.5 h-3.5 text-cyan" />
              <span>Data Export</span>
            </div>
            {openSection === 'data' ? <ChevronUp className="w-4 h-4 text-gray" /> : <ChevronDown className="w-4 h-4 text-gray" />}
          </button>

          {openSection === 'data' && (
            <div className="p-4 border-t border-white/5 flex flex-col gap-3.5 bg-navy/20 text-[10px] font-mono">
              {/* Logging toggle */}
              <div className="flex items-center justify-between text-[10px] font-semibold text-gray uppercase font-sans">
                <span>Auto-Log telemetry</span>
                <button
                  onClick={() => !isReadOnly && updateSetting('dataManagement.loggingEnabled', !settings.dataManagement.loggingEnabled)}
                  disabled={isReadOnly}
                  className="cursor-pointer focus:outline-none disabled:opacity-50"
                >
                  <div className={`w-8 h-4 rounded-full p-0.5 transition-all duration-150 ${settings.dataManagement.loggingEnabled ? 'bg-cyan' : 'bg-navy/80'}`}>
                    <div className={`w-3 h-3 rounded-full bg-white transition-all duration-150 ${settings.dataManagement.loggingEnabled ? 'translate-x-4 shadow-sm' : 'translate-x-0'}`}></div>
                  </div>
                </button>
              </div>

              {/* Log Frequency */}
              {settings.dataManagement.loggingEnabled && (
                <div className="flex items-center justify-between text-[9px] text-gray uppercase font-sans border-t border-white/5 pt-2">
                  <span>Frequency:</span>
                  <select
                    value={settings.dataManagement.logFrequency}
                    disabled={isReadOnly}
                    onChange={(e) => updateSetting('dataManagement.logFrequency', e.target.value)}
                    className="bg-navy border border-white/10 rounded px-1.5 py-1 text-[10px] text-white font-medium focus:outline-none cursor-pointer disabled:opacity-50"
                  >
                    <option value="every-frame">Every Frame</option>
                    <option value="per-second">Per Second</option>
                    <option value="per-minute">Per Minute</option>
                  </select>
                </div>
              )}

              {/* Horizontal Divider */}
              <div className="border-t border-white/5 my-0.5"></div>

              {/* Date Pickers for Export */}
              <div className="grid grid-cols-2 gap-2">
                <div className="flex flex-col">
                  <span className="text-[8px] font-bold text-gray uppercase mb-0.5 font-sans">Start Date</span>
                  <input
                    type="date"
                    value={settings.dataManagement.dateRangeStart}
                    onChange={(e) => updateSetting('dataManagement.dateRangeStart', e.target.value)}
                    className="bg-navy border border-white/10 rounded px-1.5 py-1 text-[10px] text-white font-mono cursor-pointer"
                  />
                </div>
                <div className="flex flex-col">
                  <span className="text-[8px] font-bold text-gray uppercase mb-0.5 font-sans">End Date</span>
                  <input
                    type="date"
                    value={settings.dataManagement.dateRangeEnd}
                    onChange={(e) => updateSetting('dataManagement.dateRangeEnd', e.target.value)}
                    className="bg-navy border border-white/10 rounded px-1.5 py-1 text-[10px] text-white font-mono cursor-pointer"
                  />
                </div>
              </div>

              {/* Export selection format */}
              <div className="flex items-center justify-between text-[9px] text-gray uppercase font-sans">
                <span>Report Format:</span>
                <select
                  value={settings.dataManagement.exportFormat}
                  onChange={(e) => updateSetting('dataManagement.exportFormat', e.target.value)}
                  className="bg-navy border border-white/10 rounded px-1.5 py-1 text-[10px] text-white font-medium focus:outline-none cursor-pointer"
                >
                  <option value="json">JSON Metadata</option>
                  <option value="csv">CSV Sheet</option>
                </select>
              </div>

              {/* Action buttons */}
              <div className="grid grid-cols-2 gap-2 mt-1.5">
                <button
                  onClick={exportData}
                  className="py-2 px-1 bg-cyan hover:bg-light-cyan text-black font-extrabold text-[10px] tracking-wider uppercase rounded-xl flex items-center justify-center gap-1 shadow-sm shadow-cyan/20 transition-all cursor-pointer hover:scale-105 active:scale-95"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export</span>
                </button>
                
                <button
                  onClick={clearHistory}
                  disabled={isReadOnly}
                  className="py-2 px-1 bg-red/10 hover:bg-red/20 border border-red/20 hover:border-red/50 text-red font-bold text-[10px] tracking-wider uppercase rounded-xl flex items-center justify-center gap-1 transition-all cursor-pointer hover:scale-105 active:scale-95 disabled:opacity-50"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Reset</span>
                </button>
              </div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}

export default ControlPanel;
