import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Activity, 
  Video, 
  Database,
  HardDrive,
  Clock,
  User,
  LogIn,
  LogOut
} from 'lucide-react';

function StatusBar({ wsStatus, latency, fps, systemStats, settings, userRole, onLoginClick, onLogout }) {
  const [timeStr, setTimeStr] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const d = new Date();
      const hrs = String(d.getHours()).padStart(2, '0');
      const mins = String(d.getMinutes()).padStart(2, '0');
      const secs = String(d.getSeconds()).padStart(2, '0');
      setTimeStr(`${hrs}:${mins}:${secs}`);
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  const getStatusDetails = () => {
    switch (wsStatus) {
      case 'connected': 
        return { label: 'ACTIVE', color: 'text-green', dotClass: 'bg-green shadow-sm' };
      case 'connecting': 
        return { label: 'CONNECTING', color: 'text-orange', dotClass: 'bg-orange shadow-sm' };
      case 'demo': 
        return { label: 'DEMO MODE', color: 'text-cyan', dotClass: 'bg-cyan shadow-sm' };
      default: 
        return { label: 'OFFLINE', color: 'text-red', dotClass: 'bg-red shadow-sm' };
    }
  };

  const status = getStatusDetails();
  const cam1Enabled = settings.cameraControls.cam1.enabled;

  return (
    <header className="glass-panel sticky top-0 z-50 w-full px-6 py-3 flex flex-col xl:flex-row justify-between items-start xl:items-center border-b border-white/5 bg-charcoal/90 backdrop-blur-xl shadow-lg shadow-black/40 gap-4">
      
      {/* Kizen Engineering Official Brand Header */}
      <div className="flex items-center gap-4">
        <a 
          href="https://kizen.co.in/" 
          target="_blank" 
          rel="noopener noreferrer" 
          className="flex items-center gap-3 no-underline group"
          title="Visit Kizen Engineering Official Website (kizen.co.in)"
        >
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan/20 to-blue-600/20 border border-cyan/40 flex items-center justify-center shadow-md shadow-cyan/10">
            <span className="text-cyan font-black text-xs font-mono">KZ</span>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-extrabold tracking-[0.06em] text-white uppercase font-sans">
                KIZEN <span className="text-cyan">ENGINEERING</span>
              </span>
              <span className="text-[8px] bg-cyan/15 border border-cyan/35 text-cyan font-bold px-1.5 py-0.2 rounded ml-1 tracking-wider uppercase font-mono">
                VISION AI
              </span>
            </div>
            <p className="text-[9px] text-gray uppercase tracking-widest font-bold font-mono">
              Innovation Is Our Tradition • <span className="text-cyan/80 group-hover:underline">kizen.co.in</span>
            </p>
          </div>
        </a>
      </div>

      {/* Live System Metrics Grid */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[11px] text-gray w-full xl:w-auto xl:justify-end">
        
        {/* Status Indicator */}
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${status.dotClass}`}></span>
          <span>SYSTEM:</span>
          <span className={`font-bold tracking-wider ${status.color}`}>{status.label}</span>
        </div>

        {/* Latency */}
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-cyan" />
          <span>LATENCY:</span>
          <span className="font-bold text-white">{wsStatus === 'disconnected' ? '--' : `${latency}ms`}</span>
        </div>

        {/* Inference FPS */}
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-cyan" />
          <span>INF FPS:</span>
          <span className="font-bold text-white">{wsStatus === 'disconnected' ? '0.0' : fps.toFixed(1)}</span>
        </div>

        {/* Line-Scan Slice Acquisition Rate */}
        {systemStats.slice_fps !== undefined && (
          <div className="flex items-center gap-1.5 bg-cyan/10 px-2 py-0.5 rounded border border-cyan/25">
            <Video className="w-3.5 h-3.5 text-cyan" />
            <span className="text-cyan">SCAN:</span>
            <span className="font-bold text-white">
              {systemStats.slice_fps} SLICES/S ({systemStats.line_freq_khz ? `${systemStats.line_freq_khz} kHz` : `${(systemStats.slice_fps * 256 / 1000).toFixed(1)} kHz`})
            </span>
          </div>
        )}

        {/* CPU */}
        <div className="flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5 text-cyan" />
          <span>CPU:</span>
          <span className="font-bold text-white">{wsStatus === 'disconnected' ? '--' : `${systemStats.cpu}%`}</span>
        </div>

        {/* RAM */}
        <div className="flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-light-cyan" />
          <span>RAM:</span>
          <span className="font-bold text-white">{wsStatus === 'disconnected' ? '--' : `${systemStats.ram}%`}</span>
        </div>

        {/* GPU VRAM */}
        {systemStats.vram_used_gb !== undefined && (
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-green" />
            <span>VRAM:</span>
            <span className="font-bold text-white">{systemStats.vram_used_gb}GB</span>
          </div>
        )}

        {/* Connected Cameras Dot Matrix */}
        <div className="flex items-center gap-2">
          <Video className="w-3.5 h-3.5 text-green" />
          <span>CAM:</span>
          <div className="flex gap-1 items-center bg-navy px-1.5 py-0.5 rounded border border-white/5">
            <span className={`w-1.5 h-1.5 rounded-full ${cam1Enabled ? 'bg-green animate-pulse' : 'bg-red'}`} title="Line Scan Camera"></span>
            <span className="text-[9px] text-white font-mono font-bold">LINE SCAN</span>
          </div>
        </div>

        {/* Real-time Clock */}
        <div className="flex items-center gap-1.5 border-l border-white/10 pl-4 py-0.5 text-white/90 font-bold">
          <Clock className="w-3.5 h-3.5 text-cyan" />
          <span>{timeStr || '00:00:00'}</span>
        </div>

        {/* --- USER AUTHENTICATION CONTROLS --- */}
        <div className="flex items-center gap-2 border-l border-white/10 pl-4 py-0.5">
          <User className="w-3.5 h-3.5 text-light-cyan" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-300">
            {userRole === 'operator' ? 'Operator' : userRole.toUpperCase()}
          </span>
          {userRole === 'operator' ? (
            <button
              onClick={onLoginClick}
              title="Authenticate to configure system parameters"
              className="ml-1 px-2.5 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded font-bold text-[9px] uppercase tracking-wider transition-all flex items-center gap-1 cursor-pointer"
            >
              <LogIn className="w-2.5 h-2.5" /> Login
            </button>
          ) : (
            <button
              onClick={onLogout}
              className="ml-1 px-2.5 py-1 bg-red/10 border border-red/35 hover:bg-red/20 text-red rounded font-bold text-[9px] uppercase tracking-wider transition-all flex items-center gap-1 cursor-pointer"
            >
              <LogOut className="w-2.5 h-2.5" /> Logout
            </button>
          )}
        </div>

      </div>

    </header>
  );
}

export default StatusBar;
