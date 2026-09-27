import React, { useState, useEffect } from 'react';
import { Eye, ShieldAlert, Activity, Wifi, RefreshCw, AlertCircle } from 'lucide-react';

const DEFECT_COLORS = {
  'Broken stitch': { border: '#f43f5e', fill: 'rgba(244, 63, 94, 0.15)', text: '#ffffff', label: 'Broken Stitch' },
  'hole': { border: '#ff5722', fill: 'rgba(255, 87, 34, 0.15)', text: '#ffffff', label: 'Hole' },
  'horizontal': { border: '#eab308', fill: 'rgba(234, 179, 8, 0.15)', text: '#000000', label: 'Horizontal Defect' },
  'lines': { border: '#a855f7', fill: 'rgba(168, 85, 247, 0.15)', text: '#ffffff', label: 'Lines' },
  'Needle mark': { border: '#10b981', fill: 'rgba(16, 185, 129, 0.15)', text: '#ffffff', label: 'Needle Mark' },
  'Pinched fabric': { border: '#3b82f6', fill: 'rgba(59, 130, 246, 0.15)', text: '#ffffff', label: 'Pinched Fabric' },
  'stain': { border: '#ec4899', fill: 'rgba(236, 72, 153, 0.15)', text: '#ffffff', label: 'Stain' },
  'Vertical': { border: '#06b6d4', fill: 'rgba(6, 182, 212, 0.15)', text: '#ffffff', label: 'Vertical Defect' }
};

function CameraFeed({ activeDefects = [], settings, onSelectDefect, cameraTelemetry, isCameraReconnecting = false }) {
  const [streamLoaded, setStreamLoaded] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const cam1Settings = settings?.cameraControls?.cam1 || { enabled: true };
  const cam1Defects = (activeDefects || []).filter(d => !d.camera || d.camera === 1 || d.camera === 'stitched');
  const quality1 = Math.max(70, 100 - cam1Defects.length * 8);

  const isOnline = cameraTelemetry?.cam1?.online ?? false;
  const isLineScan = cameraTelemetry?.is_linescan ?? true;
  const sliceFps = cameraTelemetry?.cam1?.slice_fps || 0;
  const stripRes = cameraTelemetry?.cam1?.strip_resolution || '640x480';
  const cameraModel = cameraTelemetry?.cam1?.camera_model || (isLineScan ? 'MindVision GigE Line-Scan' : 'Optical Sensor');

  // Reset stream error if online status recovers
  useEffect(() => {
    if (isOnline) {
      setStreamError(false);
    }
  }, [isOnline]);

  return (
    <div className="flex justify-center w-full">
      {/* Primary Line-Scan Viewport */}
      <div className={`relative w-full rounded-2xl overflow-hidden glass-panel border ${cam1Settings.enabled ? 'border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.5)]' : 'border-white/5 opacity-40'} group transition-all duration-300 hover:border-cyan/30`}>
        
        {/* Title overlay */}
        <div className="absolute top-3 left-3 z-20 flex items-center gap-2 px-3 py-1.5 bg-slate-900/90 backdrop-blur-md rounded-full border border-white/10 shadow-lg">
          <Eye className="w-3.5 h-3.5 text-cyan" />
          <span className="text-[10px] font-extrabold tracking-wider uppercase text-white font-mono">
            {cameraModel.toUpperCase()}
          </span>
          
          {isCameraReconnecting ? (
            <span className="text-[8px] font-bold px-2 py-0.5 rounded border bg-amber-500/20 text-amber-300 border-amber-500/40 flex items-center gap-1">
              <RefreshCw className="w-2.5 h-2.5 animate-spin" />
              RECONNECTING
            </span>
          ) : isOnline ? (
            <span className="text-[8px] font-bold px-2 py-0.5 rounded border bg-green/20 text-green border-green/40 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green animate-pulse"></span>
              {sliceFps > 0 ? `${sliceFps} SLICES/S • ${(sliceFps * 256 / 1000).toFixed(1)} kHz` : 'ONLINE'}
            </span>
          ) : (
            <span className="text-[8px] font-bold px-2 py-0.5 rounded border bg-red/20 text-red border-red/40 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red"></span>
              STANDBY
            </span>
          )}
        </div>

        {/* Quality / Defect Count overlay badge */}
        {cam1Settings.enabled && isOnline && (
          <div className="absolute top-3 right-3 z-20 flex items-center gap-2.5 px-3 py-1.5 bg-slate-900/90 backdrop-blur-md rounded-full border border-white/10 text-[9px] font-mono font-bold tracking-wide shadow-lg">
            <span className="text-red flex items-center gap-1">
              <Activity className="w-3 h-3" />
              DEFECTS: {cam1Defects.length}
            </span>
            <span className="text-white/20">|</span>
            <span className="text-green">GRADE: {quality1}%</span>
          </div>
        )}

        {/* Video stream container */}
        <div className="aspect-[4/3] w-full bg-slate-950 flex items-center justify-center overflow-hidden relative select-none">
          {/* Industrial Corner Calibration Markers */}
          <div className="absolute top-2 left-2 w-3.5 h-3.5 border-t-2 border-l-2 border-cyan/50 pointer-events-none z-10"></div>
          <div className="absolute top-2 right-2 w-3.5 h-3.5 border-t-2 border-r-2 border-cyan/50 pointer-events-none z-10"></div>
          <div className="absolute bottom-2 left-2 w-3.5 h-3.5 border-b-2 border-l-2 border-cyan/50 pointer-events-none z-10"></div>
          <div className="absolute bottom-2 right-2 w-3.5 h-3.5 border-b-2 border-r-2 border-cyan/50 pointer-events-none z-10"></div>
          
          {!cam1Settings.enabled ? (
            <div className="flex flex-col items-center gap-2 text-gray text-xs font-mono">
              <ShieldAlert className="w-8 h-8 opacity-45 text-red" />
              <span>SENSOR INTERFACE DISABLED</span>
            </div>
          ) : isCameraReconnecting ? (
            <div className="flex flex-col items-center gap-3 text-amber-300 text-xs font-mono p-6 text-center">
              <RefreshCw className="w-10 h-10 animate-spin text-amber-400 opacity-80" />
              <div className="font-bold tracking-widest uppercase">Hardware Reconnection Watchdog Active</div>
              <p className="text-[10px] text-slate-400 max-w-xs">
                Attempting automatic handshake with MindVision GigE line-scan receiver...
              </p>
            </div>
          ) : isOnline && !streamError ? (
            <div className="relative w-full h-full flex items-center justify-center">
              {!streamLoaded && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-950 z-10">
                  <div className="w-6 h-6 border-2 border-cyan border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-[9px] font-mono text-cyan tracking-wider uppercase">Synchronizing Stream...</span>
                </div>
              )}
              {/* Clean Video Feed without CSS filter double-processing */}
              <img
                src="/video_feed/1"
                alt="Kizen Industrial Fabric Feed"
                onLoad={() => {
                  setStreamLoaded(true);
                  setStreamError(false);
                }}
                onError={() => {
                  setStreamError(true);
                }}
                className={`w-full h-full object-cover transition-opacity duration-300 ${streamLoaded ? 'opacity-100' : 'opacity-0'}`}
              />
            </div>
          ) : (
            /* Authentic Industrial Offline Standby Slate */
            <div className="flex flex-col items-center justify-center gap-3 p-8 text-center bg-slate-950 w-full h-full">
              <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-slate-500">
                <Wifi className="w-6 h-6 opacity-60" />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs font-mono font-bold tracking-wider text-slate-300 uppercase">
                  Hardware Sensor Standby
                </span>
                <span className="text-[10px] font-mono text-slate-500 max-w-sm">
                  Connect ChinaVision GELM44M-T2 GigE Line-Scan Camera or verify physical Ethernet connection at 169.254.231.206.
                </span>
              </div>
            </div>
          )}

          {/* SVG Bounding Boxes Overlay for Real Detections */}
          {cam1Settings.enabled && isOnline && streamLoaded && (
            <svg 
              className="absolute inset-0 w-full h-full pointer-events-auto"
              viewBox="0 0 640 480"
              preserveAspectRatio="none"
            >
              {cam1Defects.map((def) => {
                const color = DEFECT_COLORS[def.type] || { border: '#E63946', fill: 'rgba(230,57,70,0.15)', text: '#ffffff', label: 'Defect' };
                const confPercent = Math.round((def.confidence || 0) * 100);
                const boxW = Math.max(12, def.bbox?.width || 20);
                const boxH = Math.max(12, def.bbox?.height || 20);
                const boxX = Math.max(0, Math.min(640 - boxW, def.bbox?.x || 0));
                const boxY = Math.max(0, Math.min(480 - boxH, def.bbox?.y || 0));
                const badgeW = Math.max(boxW, 115);
                const badgeY = boxY > 20 ? boxY - 18 : boxY + boxH + 2;

                return (
                  <g 
                    key={def.id} 
                    className="cursor-pointer group/box"
                    onClick={() => onSelectDefect && onSelectDefect(def)}
                  >
                    <rect
                      x={boxX}
                      y={boxY}
                      width={boxW}
                      height={boxH}
                      fill={color.fill}
                      stroke={color.border}
                      strokeWidth="2.5"
                      style={{ filter: `drop-shadow(0 0 6px ${color.border})` }}
                      className="transition-all duration-150 hover:stroke-white hover:stroke-[3]"
                    />
                    
                    <rect
                      x={boxX}
                      y={badgeY}
                      width={badgeW}
                      height={18}
                      fill={color.border}
                      rx={3}
                    />

                    <text
                      x={boxX + 4}
                      y={badgeY + 12}
                      fill={color.text}
                      fontSize="9"
                      fontWeight="bold"
                      fontFamily="IBM Plex Mono, monospace"
                    >
                      {color.label.toUpperCase()} {confPercent}%
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        {/* Timestamp & Feed Specs Overlay */}
        <div className="absolute bottom-3 left-3 right-3 z-20 flex justify-between items-center text-[9px] font-mono text-slate-400 font-semibold bg-slate-900/90 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10 shadow-lg">
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan"></span>
            KIZEN TEXTILEGUARD • STRIP: {stripRes}
          </span>
          <span className="text-slate-400 font-mono">TIME: {currentTime}</span>
        </div>
      </div>
    </div>
  );
}

export default CameraFeed;
