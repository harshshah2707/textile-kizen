import React, { useState, useEffect, useRef, useCallback } from 'react';
import StatusBar from './components/StatusBar';
import CameraFeed from './components/CameraFeed';
import Dashboard from './components/Dashboard';
import ControlPanel from './components/ControlPanel';
import DefectsList from './components/DefectsList';
import Analytics from './components/Analytics';
import RollVisualizer from './components/RollVisualizer';
import CuttingMapGuide from './components/CuttingMapGuide';
import SampleCollector from './components/SampleCollector';
import KizenLogo from './components/KizenLogo';
import { Play, BarChart3, Settings as SettingsIcon, Layers, ChevronLeft, ChevronRight, Scissors, Grid, Camera, LogOut, HelpCircle, Mail, Info } from 'lucide-react';

// Helper to synthesize a warning alert beep programmatically using Web Audio API
const playBeep = (freq = 800, duration = 0.15) => {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gainNode.gain.setValueAtTime(0.05, audioCtx.currentTime);
    // Beep fade out
    gainNode.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + duration);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + duration);
  } catch (error) {
    console.warn("Audio Context failed to play beep:", error);
  }
};

const DEFAULT_SETTINGS = {
  preset: 'normal',
  confidenceThreshold: 0.35,
  iouThreshold: 0.45,
  minDefectArea: 1.0,
  maxDefectArea: 500.0,
  fabricWidth: 1800, // fabric width in mm
  maxAllowedPoints: 40, // max points per 100 sq meters (ASTM D5430)
  defectFilters: {
    'Broken stitch': { enabled: true, threshold: 0.35, priority: 'critical', alert: true },
    'hole': { enabled: true, threshold: 0.35, priority: 'critical', alert: true },
    'horizontal': { enabled: true, threshold: 0.35, priority: 'high', alert: true },
    'lines': { enabled: true, threshold: 0.35, priority: 'high', alert: true },
    'Needle mark': { enabled: true, threshold: 0.30, priority: 'medium', alert: true },
    'Pinched fabric': { enabled: true, threshold: 0.30, priority: 'medium', alert: true },
    'stain': { enabled: true, threshold: 0.30, priority: 'high', alert: true },
    'Vertical': { enabled: true, threshold: 0.35, priority: 'high', alert: true }
  },
  cameraControls: {
    cam1: { enabled: true, brightness: 50, exposure: 20, contrast: 50, enhance: false, recording: false },
    cam2: { enabled: true, brightness: 50, exposure: 20, contrast: 50, enhance: false, recording: false }
  },
  alerts: {
    defectsMinThreshold: 5,
    soundEnabled: true,
    flashEnabled: true,
    email: 'quality-control@textileguard.ai',
    cooldown: 5
  },
  dataManagement: {
    loggingEnabled: true,
    logFrequency: 'per-second',
    exportFormat: 'json',
    dateRangeStart: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    dateRangeEnd: new Date().toISOString().split('T')[0]
  },
  fabricSpeed: 12.0 // m/min
};

function App() {
  const [activeTab, setActiveTab] = useState('inspection');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(true);
  const [sessionStartTime, setSessionStartTime] = useState(Date.now());

  // Load settings from LocalStorage or default
  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem('textileguard_settings');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return { ...DEFAULT_SETTINGS, ...parsed };
      } catch (e) {
        return DEFAULT_SETTINGS;
      }
    }
    return DEFAULT_SETTINGS;
  });

  const isDemoMode = false;
  const [wsStatus, setWsStatus] = useState('disconnected'); // connected, disconnected, connecting
  const [isSimulationActive, setIsSimulationActive] = useState(false);
  const [wsLatency, setWsLatency] = useState(12);
  const [fps, setFps] = useState(30);
  const [activeDefects, setActiveDefects] = useState([]);
  const [defectHistory, setDefectHistory] = useState([]);
  const [batchStats, setBatchStats] = useState({
    processedMeters: 0,
    totalDefects: 0,
    timeElapsed: 0,
    status: 'PASS'
  });
  const [selectedDefect, setSelectedDefect] = useState(null);
  const [isAlerting, setIsAlerting] = useState(false);
  const [systemStats, setSystemStats] = useState({
    cpu: 0,
    ram: 0,
    disk: 0
  });

  // User Auth & Material Presets States
  const [userRole, setUserRole] = useState('operator');
  const [materials, setMaterials] = useState([
    { name: "Denim", yolo_conf: 0.40, yolo_iou: 0.50, fabric_width: 1600, max_points_100m: 35, exposure: 15.0, gain: 24, slice_height: 250, clahe_enabled: 1 },
    { name: "Cotton", yolo_conf: 0.35, yolo_iou: 0.45, fabric_width: 1800, max_points_100m: 40, exposure: 12.0, gain: 18, slice_height: 200, clahe_enabled: 1 },
    { name: "Shirt fabric", yolo_conf: 0.30, yolo_iou: 0.40, fabric_width: 1500, max_points_100m: 30, exposure: 8.0, gain: 16, slice_height: 180, clahe_enabled: 0 },
    { name: "Linen", yolo_conf: 0.45, yolo_iou: 0.50, fabric_width: 1700, max_points_100m: 50, exposure: 20.0, gain: 32, slice_height: 300, clahe_enabled: 1 }
  ]);
  const [selectedMaterial, setSelectedMaterial] = useState('Cotton');
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  
  // Active inspection session states
  const [activeRollId, setActiveRollId] = useState(null);
  const [activeRollNumber, setActiveRollNumber] = useState(null);
  const [rollNumberInput, setRollNumberInput] = useState('');
  const [wantsSimulation, setWantsSimulation] = useState(true);
  const [operatorNameInput, setOperatorNameInput] = useState('operator');
  const [historicalRolls, setHistoricalRolls] = useState([]);

  // Boot Screen & Support States
  const [currentScreen, setCurrentScreen] = useState('boot'); // boot, app
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showInitInspectionModal, setShowInitInspectionModal] = useState(false);
  const [showContactModal, setShowContactModal] = useState(false);
  const [contactEmail, setContactEmail] = useState('');
  const [contactSubject, setContactSubject] = useState('');
  const [contactMessage, setContactMessage] = useState('');

  const wsRef = useRef(null);
  const lastAlertTimeRef = useRef(0);
  const defectHistoryRef = useRef([]);
  defectHistoryRef.current = defectHistory;

  // Auto-save settings
  useEffect(() => {
    localStorage.setItem('textileguard_settings', JSON.stringify(settings));
  }, [settings]);

  // Load materials and historical rolls on mount
  const fetchMaterials = () => {
    fetch('/api/materials')
      .then(res => res.json())
      .then(data => {
        if (data && data.length > 0) {
          setMaterials(data);
        }
      })
      .catch(err => console.warn("Failed to fetch materials list", err));
  };

  const fetchHistoricalRolls = () => {
    fetch('/api/rolls')
      .then(res => res.json())
      .then(data => {
        if (data) {
          setHistoricalRolls(data);
        }
      })
      .catch(err => console.warn("Failed to fetch rolls history", err));
  };

  useEffect(() => {
    fetchMaterials();
    fetchHistoricalRolls();
  }, []);

  const handleLoginSubmit = (e) => {
    e.preventDefault();
    setLoginError('');
    fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: loginUsername, password: loginPassword })
    })
      .then(res => {
        if (!res.ok) throw new Error("Invalid username or password");
        return res.json();
      })
      .then(data => {
        setUserRole(data.role);
        setOperatorNameInput(loginUsername);
        setIsLoggedIn(true);
        setIsLoginModalOpen(false);
        setLoginUsername('');
        setLoginPassword('');
      })
      .catch(err => {
        setLoginError(err.message);
      });
  };

  const handleLogout = () => {
    setUserRole('operator');
    setOperatorNameInput('operator');
    setIsLoggedIn(false);
    setCurrentScreen('boot');
  };

  const handleContactSubmit = (e) => {
    e.preventDefault();
    fetch('/api/support/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: contactEmail,
        subject: contactSubject,
        message: contactMessage
      })
    })
      .then(res => res.json())
      .then(data => {
        alert("Your support request has been submitted successfully to support@textileguard.ai!");
        setShowContactModal(false);
        setContactEmail('');
        setContactSubject('');
        setContactMessage('');
      })
      .catch(err => alert("Failed to send message: " + err));
  };

  const handleSelectMaterial = (name) => {
    setSelectedMaterial(name);
    const m = materials.find(x => x.name === name);
    if (m) {
      setSettings(prev => ({
        ...prev,
        confidenceThreshold: m.yolo_conf,
        iouThreshold: m.yolo_iou,
        fabricWidth: m.fabric_width,
        maxAllowedPoints: m.max_points_100m,
        cameraControls: {
          ...prev.cameraControls,
          cam1: {
            ...prev.cameraControls.cam1,
            exposure: m.exposure,
            gain: m.gain,
            slice_height: m.slice_height,
            enhance: m.clahe_enabled === 1
          }
        }
      }));
    }
  };

  const handleSavePreset = () => {
    const activePreset = materials.find(x => x.name === selectedMaterial);
    if (!activePreset) return;
    
    const payload = {
      name: selectedMaterial,
      yolo_conf: settings.confidenceThreshold,
      yolo_iou: settings.iouThreshold,
      fabric_width: settings.fabricWidth,
      max_points_100m: settings.maxAllowedPoints,
      exposure: settings.cameraControls.cam1.exposure,
      gain: settings.cameraControls.cam1.gain,
      slice_height: settings.cameraControls.cam1.slice_height,
      clahe_enabled: settings.cameraControls.cam1.enhance ? 1 : 0
    };
    
    fetch('/api/materials/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => {
        alert(`Material preset "${selectedMaterial}" settings saved successfully!`);
        fetchMaterials();
      })
      .catch(err => alert(`Failed to save preset: ${err}`));
  };

  const handleStartInspection = () => {
    const rollNum = rollNumberInput.trim() || `ROLL-${Date.now()}`;
    
    fetch('/api/rolls/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        roll_number: rollNum,
        material_name: selectedMaterial,
        operator_name: operatorNameInput
      })
    })
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success') {
          setActiveRollId(data.roll_id);
          setActiveRollNumber(data.roll_number);
          setDefectHistory([]);
          setActiveDefects([]);
          setSelectedDefect(null);
          setBatchStats({
            processedMeters: 0,
            totalDefects: 0,
            timeElapsed: 0,
            status: 'PASS'
          });
          // Synchronize Settings via WS
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'settings',
              confidenceThreshold: settings.confidenceThreshold,
              iouThreshold: settings.iouThreshold,
              fabricSpeed: settings.fabricSpeed,
              fabricWidth: settings.fabricWidth,
              cameraControls: settings.cameraControls
            }));
            wsRef.current.send(JSON.stringify({
              type: 'toggle_simulation',
              enabled: wantsSimulation
            }));
          }
        }
      })
      .catch(err => alert(`Failed to start roll: ${err}`));
  };

  const handleStopInspection = () => {
    if (!activeRollId) return;
    
    const metrics = calculate4PointMetrics(defectHistory, batchStats.processedMeters);
    
    fetch('/api/rolls/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        length_meters: batchStats.processedMeters,
        total_points: metrics.totalPenaltyPoints,
        points_per_100m: metrics.pointsPer100m2,
        grade: metrics.grade
      })
    })
      .then(res => res.json())
      .then(data => {
        alert(`Inspection finalized for Roll: ${activeRollNumber}!\nLength: ${batchStats.processedMeters.toFixed(2)}m\nGrade: ${metrics.grade}`);
        setActiveRollId(null);
        setActiveRollNumber(null);
        setRollNumberInput('');
        // Toggle simulation off
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            type: 'toggle_simulation',
            enabled: false
          }));
        }
        fetchHistoricalRolls();
      })
      .catch(err => alert(`Failed to finalize roll: ${err}`));
  };

  // Apply Presets
  const applyPreset = useCallback((presetName) => {
    let presetChanges = {};
    if (presetName === 'strict') {
      presetChanges = {
        preset: 'strict',
        confidenceThreshold: 0.50,
        minDefectArea: 0.5,
        maxDefectArea: 200.0,
        defectFilters: {
          'Broken stitch': { enabled: true, threshold: 0.50, priority: 'critical', alert: true },
          'hole': { enabled: true, threshold: 0.50, priority: 'critical', alert: true },
          'horizontal': { enabled: true, threshold: 0.50, priority: 'high', alert: true },
          'lines': { enabled: true, threshold: 0.50, priority: 'high', alert: true },
          'Needle mark': { enabled: true, threshold: 0.45, priority: 'medium', alert: true },
          'Pinched fabric': { enabled: true, threshold: 0.45, priority: 'medium', alert: true },
          'stain': { enabled: true, threshold: 0.45, priority: 'high', alert: true },
          'Vertical': { enabled: true, threshold: 0.50, priority: 'high', alert: true }
        }
      };
    } else if (presetName === 'normal') {
      presetChanges = {
        preset: 'normal',
        confidenceThreshold: 0.35,
        minDefectArea: 1.0,
        maxDefectArea: 500.0,
        defectFilters: {
          'Broken stitch': { enabled: true, threshold: 0.35, priority: 'critical', alert: true },
          'hole': { enabled: true, threshold: 0.35, priority: 'critical', alert: true },
          'horizontal': { enabled: true, threshold: 0.35, priority: 'high', alert: true },
          'lines': { enabled: true, threshold: 0.35, priority: 'high', alert: true },
          'Needle mark': { enabled: true, threshold: 0.30, priority: 'medium', alert: true },
          'Pinched fabric': { enabled: true, threshold: 0.30, priority: 'medium', alert: true },
          'stain': { enabled: true, threshold: 0.30, priority: 'high', alert: true },
          'Vertical': { enabled: true, threshold: 0.35, priority: 'high', alert: true }
        }
      };
    } else if (presetName === 'lenient') {
      presetChanges = {
        preset: 'lenient',
        confidenceThreshold: 0.25,
        minDefectArea: 3.0,
        maxDefectArea: 1000.0,
        defectFilters: {
          'Broken stitch': { enabled: true, threshold: 0.25, priority: 'high', alert: true },
          'hole': { enabled: true, threshold: 0.25, priority: 'high', alert: true },
          'horizontal': { enabled: true, threshold: 0.25, priority: 'medium', alert: false },
          'lines': { enabled: true, threshold: 0.25, priority: 'medium', alert: false },
          'Needle mark': { enabled: false, threshold: 0.20, priority: 'low', alert: false },
          'Pinched fabric': { enabled: false, threshold: 0.20, priority: 'low', alert: false },
          'stain': { enabled: true, threshold: 0.20, priority: 'medium', alert: false },
          'Vertical': { enabled: true, threshold: 0.25, priority: 'medium', alert: false }
        }
      };
    } else {
      presetChanges = { preset: 'custom' };
    }
    setSettings(prev => ({ ...prev, ...presetChanges }));
  }, []);

  // Update Settings parameter helper
  const updateSetting = useCallback((path, value) => {
    setSettings(prev => {
      const copy = { ...prev };
      const keys = path.split('.');
      let current = copy;
      for (let i = 0; i < keys.length - 1; i++) {
        current = current[keys[i]];
      }
      current[keys[keys.length - 1]] = value;
      copy.preset = 'custom'; // Any manual tweak makes it custom
      return copy;
    });
  }, []);

  // Individual defect point calculation based on ASTM D5430 (4-Point System)
  const calculateDefectPoints = useCallback((type, size_mm) => {
    if (type === 'hole') {
      return size_mm <= 25 ? 2 : 4;
    } else {
      if (size_mm <= 75) return 1;
      if (size_mm <= 150) return 2;
      if (size_mm <= 230) return 3;
      return 4;
    }
  }, []);

  // ASTM D5430 4-Point System Calculations
  const calculate4PointMetrics = useCallback((defectsList, currentMeters) => {
    const buckets = {};
    
    defectsList.forEach(d => {
      const pts = calculateDefectPoints(d.type, d.size_mm);
      const elapsedMs = d.timestamp - sessionStartTime;
      const metersFromStart = settings.fabricSpeed * (elapsedMs / 60000);
      const bucket = Math.floor(Math.max(0, metersFromStart));
      
      if (!buckets[bucket]) {
        buckets[bucket] = 0;
      }
      buckets[bucket] += pts;
    });

    let totalPenaltyPoints = 0;
    Object.keys(buckets).forEach(b => {
      totalPenaltyPoints += Math.min(4, buckets[b]);
    });

    const inspectedMeters = currentMeters !== undefined ? currentMeters : batchStats.processedMeters;
    const width_mm = settings.fabricWidth || 1800;
    
    let pointsPer100m2 = 0;
    if (inspectedMeters > 0) {
      pointsPer100m2 = (totalPenaltyPoints * 100000) / (inspectedMeters * width_mm);
    }
    pointsPer100m2 = parseFloat(pointsPer100m2.toFixed(1));

    const maxAllowed = settings.maxAllowedPoints || 40;
    const grade = pointsPer100m2 <= maxAllowed ? 'FIRST QUALITY' : 'SECOND QUALITY';
    const qualityPercentage = Math.max(0, Math.round(100 - (pointsPer100m2 / maxAllowed) * 100));

    return {
      totalPenaltyPoints,
      pointsPer100m2,
      qualityPercentage,
      grade
    };
  }, [sessionStartTime, settings.fabricSpeed, settings.fabricWidth, settings.maxAllowedPoints, batchStats.processedMeters, calculateDefectPoints]);

  // Trigger alert sound & flash
  const triggerAlert = useCallback((defect) => {
    const now = Date.now();
    const cooldownMs = settings.alerts.cooldown * 1000;
    if (now - lastAlertTimeRef.current >= cooldownMs) {
      lastAlertTimeRef.current = now;
      if (settings.alerts.soundEnabled) {
        // High priority alarm
        playBeep(900, 0.2);
        setTimeout(() => playBeep(900, 0.2), 250);
      }
      if (settings.alerts.flashEnabled) {
        setIsAlerting(true);
        setTimeout(() => setIsAlerting(false), 2000);
      }
    }
  }, [settings.alerts.soundEnabled, settings.alerts.flashEnabled, settings.alerts.cooldown]);

  // Main handler for adding new defects
  const handleNewDefect = useCallback((newDefect) => {
    // 1. Check if the type is enabled by filters
    const typeFilter = settings.defectFilters[newDefect.type];
    if (!typeFilter || !typeFilter.enabled) return;

    // 2. Check confidence threshold
    if (newDefect.confidence < settings.confidenceThreshold) return;
    if (newDefect.confidence < typeFilter.threshold) return;

    // 3. Check size constraints
    if (newDefect.size_mm < settings.minDefectArea || newDefect.size_mm > settings.maxDefectArea) return;

    // Trigger visual/auditory alerts if enabled for this filter type
    if (typeFilter.alert) {
      triggerAlert(newDefect);
    }

    // Add to history list
    setDefectHistory(prev => {
      const updated = [newDefect, ...prev];
      // Sync batch statistics
      const metrics = calculate4PointMetrics(updated);
      setBatchStats(prevStats => ({
        ...prevStats,
        totalDefects: updated.length,
        status: metrics.grade === 'FIRST QUALITY' ? 'PASS' : 'FAIL'
      }));
      return updated;
    });
  }, [settings, triggerAlert, calculate4PointMetrics]);

  // Toggle simulation mode via WebSocket
  const toggleSimulation = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        const nextState = !isSimulationActive;
        if (nextState) {
          // Reset statistics on fresh simulation run
          setDefectHistory([]);
          setActiveDefects([]);
          setSelectedDefect(null);
          setBatchStats({
            processedMeters: 0,
            totalDefects: 0,
            timeElapsed: 0,
            status: 'PASS'
          });
        }
        wsRef.current.send(JSON.stringify({
          type: 'toggle_simulation',
          enabled: nextState
        }));
      } catch (e) {
        console.warn("Failed to toggle simulation mode", e);
      }
    } else {
      alert("Loom edge server is offline. Please check connection.");
    }
  }, [isSimulationActive]);

  // WebSocket Connection Lifecycle
  useEffect(() => {
    if (isDemoMode) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setWsStatus('disconnected');
      return;
    }

    setWsStatus('connecting');
    const connectWs = () => {
      const wsUrl = `ws://${window.location.hostname}:8765`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsStatus('connected');
        console.log("WebSocket connected to Jetson");
        wsRef.current = ws;
        try {
          ws.send(JSON.stringify({
            type: "settings",
            confidenceThreshold: settings.confidenceThreshold,
            iouThreshold: settings.iouThreshold
          }));
        } catch (e) {
          console.warn("Failed to send initial settings", e);
        }
      };

      ws.onmessage = (event) => {
        try {
          const startTime = performance.now();
          const data = JSON.parse(event.data);
          
          if (data.type === 'heartbeat') {
            setSystemStats({
              cpu: data.cpu || 0,
              ram: data.ram || 0,
              disk: data.disk || 0
            });
            setWsLatency(Math.round(performance.now() - startTime));
            return;
          }

          if (data.type === 'simulation_state') {
            setIsSimulationActive(data.enabled);
            return;
          }

          if (data.type === 'frame_metrics') {
            setFps(data.fps || 30);
            return;
          }

          if (data.type === 'defect') {
            const defectObj = {
              id: data.id || Math.random().toString(36).substr(2, 9),
              type: data.defect_type,
              confidence: data.confidence,
              bbox: data.bbox, // {x, y, width, height}
              camera: data.camera || 1,
              timestamp: data.timestamp || Date.now(),
              size_mm: data.size_mm || (data.bbox.width * 0.05).toFixed(1),
              location: data.location || `${Math.round(data.bbox.x)},${Math.round(data.bbox.y)}`,
              distance_meters: data.distance_meters !== undefined ? data.distance_meters : 0.0
            };
            handleNewDefect(defectObj);
          }
        } catch (e) {
          console.error("Error reading websocket message", e);
        }
      };

      ws.onerror = (e) => {
        console.error("WebSocket error", e);
        setWsStatus('disconnected');
      };

      ws.onclose = () => {
        setWsStatus('disconnected');
        console.log("WebSocket disconnected. Retrying in 3s...");
        // Reconnect loop if not in demo mode
        if (!isDemoMode) {
          wsRef.current = setTimeout(connectWs, 3000);
        }
      };

      wsRef.current = ws;
    };

    connectWs();

    return () => {
      if (wsRef.current) {
        if (typeof wsRef.current === 'number') {
          clearTimeout(wsRef.current);
        } else {
          wsRef.current.close();
        }
      }
    };
  }, [isDemoMode, handleNewDefect]);

  // Synchronize settings with Python Backend in real-time
  useEffect(() => {
    if (wsStatus === 'connected' && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'settings',
          confidenceThreshold: settings.confidenceThreshold,
          iouThreshold: settings.iouThreshold,
          minDefectArea: settings.minDefectArea,
          maxDefectArea: settings.maxDefectArea,
          fabricSpeed: settings.fabricSpeed,
          fabricWidth: settings.fabricWidth,
          cameraControls: settings.cameraControls
        }));
      } catch (e) {
        console.warn("Failed to synchronize settings", e);
      }
    }
  }, [
    settings.confidenceThreshold, 
    settings.iouThreshold, 
    settings.minDefectArea, 
    settings.maxDefectArea, 
    settings.fabricSpeed,
    settings.fabricWidth,
    settings.cameraControls,
    wsStatus
  ]);

  // Live Loom Length & Timer tracking (only when simulation is active or in demo mode)
  useEffect(() => {
    if (wsStatus !== 'connected' && !isDemoMode) {
      return;
    }
    if (!isSimulationActive && !isDemoMode) {
      return;
    }
    const metersInterval = setInterval(() => {
      setBatchStats(prev => {
        const addedMeters = (settings.fabricSpeed / 60) * 1; // 1 second increments
        return {
          ...prev,
          processedMeters: parseFloat((prev.processedMeters + addedMeters).toFixed(2)),
          timeElapsed: prev.timeElapsed + 1
        };
      });
    }, 1000);

    return () => {
      clearInterval(metersInterval);
    };
  }, [settings.fabricSpeed, wsStatus, isDemoMode, isSimulationActive]);

  const clearHistory = useCallback(() => {
    if (window.confirm("Are you sure you want to clear the defect history? This will reset all current session batch metrics.")) {
      setDefectHistory([]);
      setActiveDefects([]);
      setSelectedDefect(null);
      setBatchStats({
        processedMeters: 0,
        totalDefects: 0,
        timeElapsed: 0,
        status: 'PASS'
      });
    }
  }, []);

  if (currentScreen === 'boot') {
    return (
      <div className="relative w-screen min-h-screen overflow-y-auto bg-navy text-white flex flex-col items-center justify-center p-6 font-sans select-none py-12">
        {/* Background Grid */}
        <div className="absolute inset-0 opacity-[0.03] bg-[radial-gradient(#ffffff_1px,transparent_1px)] [background-size:16px_16px] pointer-events-none"></div>
        
        {/* Contact/Support Modal Overlay */}
        {showContactModal && (
          <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="glass-panel w-full max-w-md p-6 rounded-2xl border border-white/10 bg-slate-900 shadow-2xl flex flex-col gap-4">
              <div className="flex justify-between items-center border-b border-white/5 pb-2">
                <h3 className="text-sm font-extrabold uppercase tracking-widest text-cyan flex items-center gap-2">
                  <Mail className="w-4 h-4" /> Send Email Ticket
                </h3>
                <button 
                  onClick={() => setShowContactModal(false)}
                  className="text-xs text-gray hover:text-white cursor-pointer"
                >
                  Cancel
                </button>
              </div>
              
              <form onSubmit={handleContactSubmit} className="flex flex-col gap-3.5 font-mono text-[10px]">
                <div className="flex flex-col gap-1">
                  <label className="text-gray uppercase font-sans font-bold">Email Address</label>
                  <input
                    type="email" required
                    value={contactEmail} onChange={(e) => setContactEmail(e.target.value)}
                    placeholder="email@example.com"
                    className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                  />
                </div>
                
                <div className="flex flex-col gap-1">
                  <label className="text-gray uppercase font-sans font-bold">Subject</label>
                  <input
                    type="text" required
                    value={contactSubject} onChange={(e) => setContactSubject(e.target.value)}
                    placeholder="System feedback, model issue, etc."
                    className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                  />
                </div>
                
                <div className="flex flex-col gap-1">
                  <label className="text-gray uppercase font-sans font-bold">Message Details</label>
                  <textarea
                    required rows={4}
                    value={contactMessage} onChange={(e) => setContactMessage(e.target.value)}
                    placeholder="Type details here..."
                    className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs font-mono"
                  />
                </div>
                
                <button
                  type="submit"
                  className="mt-2 w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-extrabold tracking-wider uppercase rounded-xl shadow-md cursor-pointer transition-all hover:scale-103"
                >
                  Send Support Email
                </button>
              </form>
            </div>
          </div>
        )}

        {/* Start Inspection Run Modal Overlay */}
        {showInitInspectionModal && (
          <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="glass-panel w-full max-w-md p-6 rounded-2xl border border-white/10 bg-slate-900 shadow-2xl flex flex-col gap-4">
              <div className="flex justify-between items-center border-b border-white/5 pb-2">
                <h3 className="text-sm font-extrabold uppercase tracking-widest text-cyan flex items-center gap-2">
                  <Play className="w-4 h-4" /> Load Inspection Parameters
                </h3>
                <button 
                  onClick={() => setShowInitInspectionModal(false)}
                  className="text-xs text-gray hover:text-white cursor-pointer"
                >
                  Cancel
                </button>
              </div>
              
              <div className="flex flex-col gap-3.5 font-mono text-[10px]">
                <div className="flex flex-col gap-1">
                  <label className="text-gray uppercase font-sans font-bold">Roll Reference Code</label>
                  <input
                    type="text"
                    value={rollNumberInput}
                    onChange={(e) => setRollNumberInput(e.target.value)}
                    placeholder="e.g. DENIM-R4001 (Auto-gen if empty)"
                    className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                  />
                </div>
                
                <div className="flex flex-col gap-1">
                  <label className="text-gray uppercase font-sans font-bold">Material Preset (Loads correct Model parameters)</label>
                  <select
                    value={selectedMaterial}
                    onChange={(e) => handleSelectMaterial(e.target.value)}
                    className="w-full px-3 py-2.5 bg-navy border border-white/10 rounded-xl text-white focus:outline-none text-xs cursor-pointer"
                  >
                    {materials.map(m => (
                      <option key={m.name} value={m.name}>{m.name}</option>
                    ))}
                  </select>
                </div>

                {/* Simulation vs Live Feed Toggle Checkbox */}
                <div 
                  className="flex items-center gap-2.5 bg-navy/60 p-3 rounded-xl border border-white/5 cursor-pointer select-none" 
                  onClick={() => setWantsSimulation(!wantsSimulation)}
                >
                  <input 
                    type="checkbox" 
                    checked={wantsSimulation}
                    onChange={(e) => setWantsSimulation(e.target.checked)}
                    className="w-4 h-4 rounded border-white/10 bg-navy cursor-pointer accent-blue-600 shrink-0"
                    onClick={(e) => e.stopPropagation()} // Prevent double trigger
                  />
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-white uppercase tracking-wider">Run in simulation mode</span>
                    <span className="text-[8px] text-gray uppercase font-sans">Simulate scanning from dataset rather than webcam</span>
                  </div>
                </div>

                <div className="bg-blue-600/10 border border-blue-500/20 p-3.5 rounded-xl text-slate-300 text-[9px] font-sans flex items-start gap-2.5">
                  <Info className="w-4 h-4 text-cyan shrink-0 mt-0.5" />
                  <div className="flex flex-col gap-0.5">
                    <span className="font-bold text-white uppercase text-[8px] tracking-wider">Model Safety Check</span>
                    <span>Selecting the correct material preset ensures the YOLO network confidence boundaries and camera exposure levels match your fabric specification.</span>
                  </div>
                </div>
                
                <button
                  onClick={() => {
                    handleStartInspection();
                    setShowInitInspectionModal(false);
                    setCurrentScreen('app');
                    setActiveTab('inspection');
                  }}
                  className="mt-2 w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-extrabold tracking-wider uppercase rounded-xl shadow-md cursor-pointer transition-all hover:scale-103"
                >
                  Start Inspection Run
                </button>
              </div>
            </div>
          </div>
        )}

        {/* KIZEN LOGO WITH ANIMATION */}
        <div className="flex flex-col items-center mb-10 text-center animate-fade-in">
          <KizenLogo className="h-28 w-auto hover:scale-102 transition-transform duration-300" lightText={true} />
          <div className="h-1.5 w-48 bg-white/5 rounded-full overflow-hidden mt-6 relative border border-white/5">
            <div className="absolute top-0 bottom-0 left-0 bg-blue-600 animate-loading-bar rounded-full"></div>
          </div>
          <p className="text-[9px] text-gray uppercase tracking-widest font-bold mt-2.5">
            Textile Guard AI Quality Control Platform
          </p>
        </div>

        {/* SCREEN 1: LOGIN REQUIRED */}
        {!isLoggedIn ? (
          <div className="glass-panel w-full max-w-sm p-6 rounded-2xl border border-white/10 bg-slate-900/60 shadow-2xl flex flex-col gap-4">
            <div className="flex flex-col items-center gap-1 border-b border-white/5 pb-3">
              <h3 className="text-xs font-black uppercase tracking-widest text-cyan">System Authorization Required</h3>
              <p className="text-[9px] text-gray uppercase tracking-wider font-semibold">Enter operator or manager credentials</p>
            </div>
            
            {loginError && (
              <div className="text-[10px] text-red bg-red/10 border border-red/25 px-3 py-1.5 rounded-lg text-center font-bold">
                ⚠️ {loginError}
              </div>
            )}
            
            <form onSubmit={handleLoginSubmit} className="flex flex-col gap-3.5 font-mono text-[10px]">
              <div className="flex flex-col gap-1">
                <label className="text-gray uppercase font-sans font-bold">Username</label>
                <input
                  type="text" required
                  value={loginUsername} onChange={(e) => setLoginUsername(e.target.value)}
                  placeholder="admin, manager, or operator"
                  className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                />
              </div>
              
              <div className="flex flex-col gap-1">
                <label className="text-gray uppercase font-sans font-bold">Password</label>
                <input
                  type="password" required
                  value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                />
              </div>
              
              <button
                type="submit"
                className="mt-2 w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-extrabold tracking-wider uppercase rounded-xl shadow-md cursor-pointer transition-all hover:scale-103"
              >
                Sign In
              </button>
            </form>
          </div>
        ) : (
          /* SCREEN 2: MAIN BOOT MENU OPTIONS */
          <div className="w-full max-w-4xl flex flex-col gap-6">
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
              
              {/* Option 1: Collect cloth sample */}
              <button
                onClick={() => {
                  setCurrentScreen('app');
                  setActiveTab('collect-samples');
                }}
                className="glass-panel p-6 rounded-2xl border border-white/5 hover:border-cyan/50 text-left transition-all hover:scale-105 active:scale-95 group flex flex-col gap-3 cursor-pointer"
              >
                <div className="bg-cyan/10 p-3 rounded-xl border border-cyan/20 w-fit text-cyan group-hover:bg-cyan group-hover:text-white transition-colors">
                  <Camera className="w-6 h-6" />
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white uppercase font-sans">1. Collect Cloth Sample</span>
                  <span className="text-[10px] text-slate-400 font-sans mt-1">Capture, slice and store fabric frames for building custom ML training datasets.</span>
                </div>
              </button>

              {/* Option 2: Defect Detection Dashboard */}
              <button
                onClick={() => {
                  setShowInitInspectionModal(true);
                }}
                className="glass-panel p-6 rounded-2xl border border-white/5 hover:border-cyan/50 text-left transition-all hover:scale-105 active:scale-95 group flex flex-col gap-3 cursor-pointer"
              >
                <div className="bg-cyan/10 p-3 rounded-xl border border-cyan/20 w-fit text-cyan group-hover:bg-cyan group-hover:text-white transition-colors">
                  <Play className="w-6 h-6" />
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white uppercase font-sans">2. Defect Detection Dashboard</span>
                  <span className="text-[10px] text-slate-400 font-sans mt-1">Launch real-time loom camera inspection streams and apply YOLO defect grading.</span>
                </div>
              </button>

              {/* Option 3: Reports History */}
              <button
                onClick={() => {
                  setCurrentScreen('app');
                  setActiveTab('cutting-map');
                }}
                className="glass-panel p-6 rounded-2xl border border-white/5 hover:border-cyan/50 text-left transition-all hover:scale-105 active:scale-95 group flex flex-col gap-3 cursor-pointer"
              >
                <div className="bg-cyan/10 p-3 rounded-xl border border-cyan/20 w-fit text-cyan group-hover:bg-cyan group-hover:text-white transition-colors">
                  <Scissors className="w-6 h-6" />
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white uppercase font-sans">3. Roll Inspection Reports</span>
                  <span className="text-[10px] text-slate-400 font-sans mt-1">See saved fabric run records, statistics, and interactive 2D cutting guides.</span>
                </div>
              </button>

              {/* Option 4: Software Performance Score */}
              <button
                onClick={() => {
                  setCurrentScreen('app');
                  setActiveTab('analytics');
                }}
                className="glass-panel p-6 rounded-2xl border border-white/5 hover:border-cyan/50 text-left transition-all hover:scale-105 active:scale-95 group flex flex-col gap-3 cursor-pointer"
              >
                <div className="bg-cyan/10 p-3 rounded-xl border border-cyan/20 w-fit text-cyan group-hover:bg-cyan group-hover:text-white transition-colors">
                  <BarChart3 className="w-6 h-6" />
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white uppercase font-sans">4. Software Quality Score</span>
                  <span className="text-[10px] text-slate-400 font-sans mt-1">Check cumulative points, metrics trends, and overall factory quality scorecards.</span>
                </div>
              </button>

              {/* Option 5: Contact us */}
              <button
                onClick={() => setShowContactModal(true)}
                className="glass-panel p-6 rounded-2xl border border-white/5 hover:border-cyan/50 text-left transition-all hover:scale-105 active:scale-95 group flex flex-col gap-3 cursor-pointer md:col-span-2 xl:col-span-1"
              >
                <div className="bg-cyan/10 p-3 rounded-xl border border-cyan/20 w-fit text-cyan group-hover:bg-cyan group-hover:text-white transition-colors">
                  <Mail className="w-6 h-6" />
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white uppercase font-sans">5. Contact Support</span>
                  <span className="text-[10px] text-slate-400 font-sans mt-1">Send a direct message or feedback report to support@textileguard.ai.</span>
                </div>
              </button>

            </div>

            {/* Operator bottom info */}
            <div className="flex justify-between items-center bg-navy/20 p-4 border border-white/5 rounded-2xl mt-4 text-[10px] font-mono text-slate-400">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-green shadow-sm"></span>
                <span>Logged in as: <strong className="text-white uppercase">{userRole}</strong></span>
              </div>
              <button
                onClick={handleLogout}
                className="px-3 py-1 bg-red/10 border border-red/20 hover:bg-red/20 text-red rounded-lg font-bold uppercase transition cursor-pointer flex items-center gap-1"
              >
                <LogOut className="w-3.5 h-3.5" /> Logout
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`flex h-screen w-screen overflow-hidden bg-navy text-white transition-colors duration-300 ${isAlerting ? 'alert-flash' : ''}`}>
      
      {/* Sidebar Navigation */}
      <aside className={`bg-slate-900 text-white flex flex-col border-r border-white/5 z-15 transition-all duration-300 ${
        isSidebarCollapsed ? 'w-16' : 'w-64'
      }`}>
        
        {/* App Title Header */}
        <div className={`p-4 border-b border-white/5 flex items-center justify-between gap-2.5 ${
          isSidebarCollapsed ? 'justify-center' : ''
        }`}>
          {!isSidebarCollapsed && (
            <div className="flex items-center gap-2.5 overflow-hidden animate-fade-in py-1">
              <KizenLogo className="h-8 w-auto" lightText={true} />
            </div>
          )}
          {isSidebarCollapsed && (
            <div className="cursor-pointer" onClick={() => setIsSidebarCollapsed(false)}>
              <KizenLogo className="h-8 w-8" lightText={true} gearOnly={true} />
            </div>
          )}
          
          {/* Toggle Button */}
          {!isSidebarCollapsed && (
            <button 
              onClick={() => setIsSidebarCollapsed(true)}
              className="p-1.5 hover:bg-white/5 rounded text-gray hover:text-white transition cursor-pointer"
            >
              <ChevronLeft size={16} />
            </button>
          )}
        </div>

        {/* Navigation Tabs */}
        <nav className="flex-1 px-2.5 py-6 flex flex-col gap-1.5">
          {/* Main Menu Button */}
          <button
            onClick={() => setCurrentScreen('boot')}
            title="Main Menu Screen"
            className={`flex items-center rounded text-xs font-semibold tracking-wide transition cursor-pointer mb-2 border-b border-white/5 pb-2 text-slate-300 hover:text-white ${
              isSidebarCollapsed ? 'justify-center p-3' : 'gap-3.5 px-4 py-3'
            }`}
          >
            <Grid size={16} className="text-cyan" />
            {!isSidebarCollapsed && <span className="truncate font-bold">Main Menu</span>}
          </button>

          <button
            onClick={() => setActiveTab("inspection")}
            title="Live Inspection"
            className={`flex items-center rounded text-xs font-semibold tracking-wide transition cursor-pointer ${
              isSidebarCollapsed ? 'justify-center p-3' : 'gap-3.5 px-4 py-3'
            } ${
              activeTab === "inspection"
                ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                : "text-gray hover:bg-white/5 hover:text-white"
            }`}
          >
            <Play size={16} />
            {!isSidebarCollapsed && <span className="truncate">Live Inspection</span>}
          </button>

          <button
            onClick={() => setActiveTab("collect-samples")}
            title="Collect Sample"
            className={`flex items-center rounded text-xs font-semibold tracking-wide transition cursor-pointer ${
              isSidebarCollapsed ? 'justify-center p-3' : 'gap-3.5 px-4 py-3'
            } ${
              activeTab === "collect-samples"
                ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                : "text-gray hover:bg-white/5 hover:text-white"
            }`}
          >
            <Camera size={16} />
            {!isSidebarCollapsed && <span className="truncate">Collect Sample</span>}
          </button>
          
          <button
            onClick={() => setActiveTab("analytics")}
            title="Analytics & Stats"
            className={`flex items-center rounded text-xs font-semibold tracking-wide transition cursor-pointer ${
              isSidebarCollapsed ? 'justify-center p-3' : 'gap-3.5 px-4 py-3'
            } ${
              activeTab === "analytics"
                ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                : "text-gray hover:bg-white/5 hover:text-white"
            }`}
          >
            <BarChart3 size={16} />
            {!isSidebarCollapsed && <span className="truncate">Analytics & Stats</span>}
          </button>

          <button
            onClick={() => setActiveTab("cutting-map")}
            title="Cutting Map Guide"
            className={`flex items-center rounded text-xs font-semibold tracking-wide transition cursor-pointer ${
              isSidebarCollapsed ? 'justify-center p-3' : 'gap-3.5 px-4 py-3'
            } ${
              activeTab === "cutting-map"
                ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                : "text-gray hover:bg-white/5 hover:text-white"
            }`}
          >
            <Scissors size={16} />
            {!isSidebarCollapsed && <span className="truncate">Cutting Pattern</span>}
          </button>

          <button
            onClick={() => setActiveTab("settings")}
            title="QC Configuration"
            className={`flex items-center rounded text-xs font-semibold tracking-wide transition cursor-pointer ${
              isSidebarCollapsed ? 'justify-center p-3' : 'gap-3.5 px-4 py-3'
            } ${
              activeTab === "settings"
                ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                : "text-gray hover:bg-white/5 hover:text-white"
            }`}
          >
            <SettingsIcon size={16} />
            {!isSidebarCollapsed && <span className="truncate">QC Configuration</span>}
          </button>
        </nav>

        {/* Expand Button at Bottom if Collapsed */}
        {isSidebarCollapsed && (
          <button 
            onClick={() => setIsSidebarCollapsed(false)}
            title="Expand Sidebar"
            className="p-3 mx-auto mb-4 hover:bg-white/5 rounded-full text-gray hover:text-white transition cursor-pointer"
          >
            <ChevronRight size={18} />
          </button>
        )}

        {/* Footer */}
        {!isSidebarCollapsed && (
          <div className="p-4 border-t border-white/5 text-[9px] text-gray uppercase tracking-widest font-mono text-center overflow-hidden animate-fade-in truncate">
            v4.5 Enterprise | PC Edge
          </div>
        )}
      </aside>

      {/* Main Workspace Area */}
      <main className="flex-1 flex flex-col overflow-hidden relative bg-navy/40">
        
        {/* Unified Top Bar Header */}
        <StatusBar
          wsStatus={isDemoMode ? 'demo' : wsStatus}
          latency={wsLatency}
          fps={fps}
          systemStats={systemStats}
          settings={settings}
          userRole={userRole}
          onLoginClick={() => setIsLoginModalOpen(true)}
          onLogout={handleLogout}
        />

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === "inspection" && (
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
              {/* LEFT COLUMN: CAMERA FEED & RECENT DEFECTS (Col-span 8) */}
              <div className="xl:col-span-8 flex flex-col gap-6">
                <CameraFeed 
                  activeDefects={activeDefects} 
                  settings={settings}
                  isDemoMode={isDemoMode}
                  onSelectDefect={setSelectedDefect}
                />
                
                <DefectsList 
                  defectHistory={defectHistory} 
                  selectedDefect={selectedDefect} 
                  onSelectDefect={setSelectedDefect}
                  calculateDefectPoints={calculateDefectPoints}
                />
              </div>

              {/* RIGHT COLUMN: GRADING SUMMARY & SESSION CONTROLS (Col-span 4) */}
              <div className="xl:col-span-4 flex flex-col gap-6">
                
                {/* Inspection Session Manager */}
                <div className="glass-panel p-5 rounded-2xl border border-white/5 shadow-md flex flex-col gap-4">
                  <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
                    <Layers className="w-3.5 h-3.5 text-cyan" />
                    <span>Roll Inspection Manager</span>
                  </h3>
                  
                  {activeRollId === null ? (
                    <div className="flex flex-col gap-3 font-mono text-[10px]">
                      <div className="flex flex-col gap-1.5">
                        <label className="text-gray uppercase font-sans font-bold">Roll Reference Code</label>
                        <input
                          type="text"
                          value={rollNumberInput}
                          onChange={(e) => setRollNumberInput(e.target.value)}
                          placeholder="e.g. DENIM-R4001 (Auto-gen if empty)"
                          className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                        />
                      </div>
                      
                      <div className="flex flex-col gap-1.5">
                        <label className="text-gray uppercase font-sans font-bold">Material Preset</label>
                        <select
                          value={selectedMaterial}
                          onChange={(e) => handleSelectMaterial(e.target.value)}
                          className="w-full px-3 py-2 bg-navy border border-white/10 rounded-xl text-white focus:outline-none text-xs cursor-pointer"
                        >
                          {materials.map(m => (
                            <option key={m.name} value={m.name}>{m.name}</option>
                          ))}
                        </select>
                      </div>
                      
                      <button
                        onClick={handleStartInspection}
                        className="mt-2 w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-extrabold tracking-wider uppercase rounded-xl shadow-md cursor-pointer transition-all hover:scale-103 text-[10px]"
                      >
                        Start Loom Roll Inspection
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-3 font-mono text-[10px]">
                      <div className="bg-blue-600/10 border border-blue-500/25 p-3 rounded-xl flex flex-col gap-1 text-[9px]">
                        <div><strong className="text-slate-300">Roll Number:</strong> <span className="text-white text-xs font-bold font-mono">{activeRollNumber}</span></div>
                        <div><strong className="text-slate-300">Material:</strong> <span className="text-slate-200">{selectedMaterial}</span></div>
                        <div><strong className="text-slate-300">Inspector:</strong> <span className="text-slate-200">{operatorNameInput}</span></div>
                      </div>
                      
                      <button
                        onClick={handleStopInspection}
                        className="w-full py-3 bg-red hover:bg-red/80 text-white font-extrabold tracking-wider uppercase rounded-xl shadow-md cursor-pointer transition-all hover:scale-103 text-[10px]"
                      >
                        Stop & Finalize Roll
                      </button>
                    </div>
                  )}
                </div>

                {/* Roll Defect Visualizer */}
                <RollVisualizer 
                  defectHistory={defectHistory}
                  processedMeters={batchStats.processedMeters}
                  selectedDefect={selectedDefect}
                  onSelectDefect={setSelectedDefect}
                />

                <Dashboard 
                  batchStats={batchStats} 
                  metrics={calculate4PointMetrics(defectHistory, batchStats.processedMeters)}
                  systemStats={systemStats}
                  wsStatus={isDemoMode ? 'demo' : wsStatus}
                />
              </div>
            </div>
          )}

          {activeTab === "collect-samples" && (
            <div className="w-full flex flex-col gap-6">
              <SampleCollector />
            </div>
          )}

          {activeTab === "analytics" && (
            <div className="w-full flex flex-col gap-6">
              <Analytics 
                defectHistory={defectHistory}
                batchStats={batchStats}
                metrics={calculate4PointMetrics(defectHistory, batchStats.processedMeters)}
                settings={settings}
              />
            </div>
          )}

          {activeTab === "cutting-map" && (
            <div className="w-full flex flex-col gap-6">
              <CuttingMapGuide 
                selectedRollId={activeRollId}
                historicalRolls={historicalRolls}
              />
            </div>
          )}

          {activeTab === "settings" && (
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 max-w-7xl mx-auto items-start">
              
              {/* Configuration panel (Col-span 4) */}
              <div className="xl:col-span-4">
                <ControlPanel
                  settings={settings}
                  updateSetting={updateSetting}
                  applyPreset={applyPreset}
                  clearHistory={clearHistory}
                  defectHistory={defectHistory}
                  isSimulationActive={isSimulationActive}
                  toggleSimulation={toggleSimulation}
                  userRole={userRole}
                  materials={materials}
                  selectedMaterial={selectedMaterial}
                  onSelectMaterial={handleSelectMaterial}
                  onSavePreset={handleSavePreset}
                />
              </div>
              
              {/* History Roll Log Table (Col-span 8) */}
              <div className="xl:col-span-8 flex flex-col gap-6">
                <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4 text-white">
                  <h3 className="text-xs font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
                    <Layers className="w-4 h-4 text-cyan" />
                    <span>Inspection Roll History Log</span>
                  </h3>
                  
                  {historicalRolls.length === 0 ? (
                    <div className="text-[10px] text-gray uppercase tracking-wider font-mono text-center py-10">
                      No historical roll sessions logged.
                    </div>
                  ) : (
                    <div className="overflow-x-auto text-[10px] font-mono">
                      <table className="w-full border-collapse">
                        <thead>
                          <tr className="border-b border-white/10 text-gray text-left">
                            <th className="py-2 pr-2 uppercase font-sans font-bold">Roll Number</th>
                            <th className="py-2 px-2 uppercase font-sans font-bold">Material</th>
                            <th className="py-2 px-2 uppercase font-sans font-bold">Operator</th>
                            <th className="py-2 px-2 uppercase font-sans font-bold">Length</th>
                            <th className="py-2 px-2 uppercase font-sans font-bold">Points</th>
                            <th className="py-2 px-2 uppercase font-sans font-bold">Grade</th>
                            <th className="py-2 pl-2 uppercase font-sans font-bold text-right">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {historicalRolls.map(r => (
                            <tr key={r.id} className="border-b border-white/5 hover:bg-white/[0.02]">
                              <td className="py-2.5 pr-2 font-bold text-white">{r.roll_number}</td>
                              <td className="py-2.5 px-2 text-slate-300">{r.material_name}</td>
                              <td className="py-2.5 px-2 text-slate-400">{r.operator_name}</td>
                              <td className="py-2.5 px-2 text-white">{r.length_meters.toFixed(2)} m</td>
                              <td className="py-2.5 px-2 text-orange font-bold">{r.total_points} pts</td>
                              <td className={`py-2.5 px-2 font-extrabold ${r.grade === 'FIRST QUALITY' ? 'text-green' : 'text-red'}`}>
                                {r.grade}
                              </td>
                              <td className="py-2.5 pl-2 text-right">
                                <a
                                  href={`/api/rolls/report/${r.id}/pdf`}
                                  className="inline-block px-3 py-1.5 bg-cyan text-black hover:bg-light-cyan font-bold rounded-lg uppercase tracking-wider text-[9px] transition-all cursor-pointer font-sans"
                                >
                                  Download PDF
                                </a>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

      </main>

      {/* Escalated Privilege Login Modal (Inside App) */}
      {isLoginModalOpen && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-panel w-full max-w-sm p-6 rounded-2xl border border-white/10 bg-slate-900 shadow-2xl flex flex-col gap-4">
            <div className="flex justify-between items-center border-b border-white/5 pb-2">
              <h3 className="text-xs font-extrabold uppercase tracking-widest text-cyan">Authenticate Manager</h3>
              <button 
                onClick={() => setIsLoginModalOpen(false)}
                className="text-xs text-gray hover:text-white cursor-pointer"
              >
                Cancel
              </button>
            </div>
            
            {loginError && (
              <div className="text-[10px] text-red bg-red/10 border border-red/25 px-3 py-1.5 rounded-lg text-center font-bold">
                ⚠️ {loginError}
              </div>
            )}
            
            <form onSubmit={handleLoginSubmit} className="flex flex-col gap-3.5 font-mono text-[10px]">
              <div className="flex flex-col gap-1">
                <label className="text-gray uppercase font-sans font-bold">Username</label>
                <input
                  type="text" required
                  value={loginUsername} onChange={(e) => setLoginUsername(e.target.value)}
                  placeholder="admin or manager"
                  className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                />
              </div>
              
              <div className="flex flex-col gap-1">
                <label className="text-gray uppercase font-sans font-bold">Password</label>
                <input
                  type="password" required
                  value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-3 py-2 bg-navy/60 border border-white/10 rounded-xl text-white focus:outline-none focus:border-cyan text-xs"
                />
              </div>
              
              <button
                type="submit"
                className="mt-2 w-full py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-extrabold tracking-wider uppercase rounded-xl shadow-md cursor-pointer transition-all hover:scale-103"
              >
                Login
              </button>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
