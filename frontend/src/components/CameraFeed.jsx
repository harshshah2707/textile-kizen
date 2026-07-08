import React, { useEffect, useRef } from 'react';
import { Eye, ShieldAlert } from 'lucide-react';

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

function CameraFeed({ activeDefects, settings, isDemoMode, onSelectDefect }) {
  const canvasRef1 = useRef(null);
  const scrollOffset1 = useRef(0);
  const animFrameId = useRef(null);

  const cam1Settings = settings.cameraControls.cam1;

  // Filter defects by camera (we only have camera 1 now)
  const cam1Defects = activeDefects.filter(d => d.camera === 1);

  // Calculate live quality index based on defect counts
  const quality1 = Math.max(70, 100 - cam1Defects.length * 8);

  // Fabric simulation loop for Demo Mode (or fallback background texture)
  useEffect(() => {
    if (!isDemoMode) return;

    const canvas1 = canvasRef1.current;
    if (!canvas1) return;

    const ctx1 = canvas1.getContext('2d');

    const drawFabric = (ctx, canvas, scrollOffset, cameraNum, activeDefs) => {
      const width = canvas.width;
      const height = canvas.height;

      // Base Fabric color (Light Blue-Grey weave)
      ctx.fillStyle = '#E6ECEF';
      ctx.fillRect(0, 0, width, height);

      // Draw vertical threads
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.04)';
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 8) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      // Draw scrolling horizontal threads
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.03)';
      const threadSpacing = 10;
      const startY = scrollOffset % threadSpacing;
      for (let y = startY; y < height; y += threadSpacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Render overlay technical grid lines
      ctx.strokeStyle = 'rgba(0, 184, 212, 0.05)';
      ctx.lineWidth = 1;
      // Horizontal grid
      for (let y = 60; y < height; y += 60) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }
      // Vertical grid
      for (let x = 80; x < width; x += 80) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }

      // Draw physical defects scrolling on canvas
      activeDefs.forEach(def => {
        if (def.camera !== cameraNum) return;

        const { x, y, width: dw, height: dh } = def.bbox;
        ctx.save();

        if (def.type === 'hole') {
          ctx.fillStyle = '#050505';
          ctx.strokeStyle = 'rgba(255, 87, 34, 0.4)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.ellipse(x + dw/2, y + dh/2, dw/2, dh/2, Math.PI/12, 0, 2*Math.PI);
          ctx.fill();
          ctx.stroke();
        } else if (def.type === 'stain') {
          const grad = ctx.createRadialGradient(x + dw/2, y + dh/2, 2, x + dw/2, y + dh/2, dw/2);
          grad.addColorStop(0, 'rgba(236, 72, 153, 0.25)');
          grad.addColorStop(0.7, 'rgba(236, 72, 153, 0.08)');
          grad.addColorStop(1, 'rgba(236, 72, 153, 0)');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.ellipse(x + dw/2, y + dh/2, dw/2, dh/2, 0, 0, 2 * Math.PI);
          ctx.fill();
        } else {
          ctx.strokeStyle = 'rgba(0, 184, 212, 0.5)';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(x, y + dh/2);
          ctx.lineTo(x + dw, y + dh/2);
          ctx.stroke();
        }

        ctx.restore();
      });
    };

    const render = () => {
      const speedFactor = settings.fabricSpeed * 0.3;
      
      if (cam1Settings.enabled) {
        scrollOffset1.current += speedFactor;
        drawFabric(ctx1, canvas1, scrollOffset1.current, 1, activeDefects);
      }

      animFrameId.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameId.current) {
        cancelAnimationFrame(animFrameId.current);
      }
    };
  }, [isDemoMode, activeDefects, settings.fabricSpeed, cam1Settings.enabled]);

  const getFilterStyle = (camSettings) => {
    if (!camSettings.enabled) return {};
    const brightness = camSettings.brightness * 2;
    const contrast = camSettings.contrast * 2;
    const enhance = camSettings.enhance ? 'contrast(1.3) saturate(1.2)' : '';
    return {
      filter: `brightness(${brightness}%) contrast(${contrast}%) ${enhance}`
    };
  };

  return (
    <div className="flex justify-center w-full">
      
      {/* Camera 1 Viewport */}
      <div className={`relative w-full rounded-2xl overflow-hidden glass-panel border ${cam1Settings.enabled ? 'border-white/5 shadow-[0_4px_24px_rgba(0,0,0,0.4)]' : 'border-white/5 opacity-40'} group transition-all duration-300 hover:scale-[1.01] hover:border-cyan/20`}>
        {/* Title overlay */}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-3 py-1 bg-charcoal/85 backdrop-blur-md rounded-full border border-white/10">
          <Eye className="w-3.5 h-3.5 text-cyan" />
          <span className="text-[10px] font-bold tracking-wider uppercase text-white font-mono">LINE SCAN CAM // LIVE INSPECTION</span>
        </div>

        {/* Quality / Defect Count overlay badge */}
        {cam1Settings.enabled && (
          <div className="absolute top-3 right-3 z-10 flex items-center gap-2.5 px-3 py-1 bg-charcoal/85 backdrop-blur-md rounded-full border border-white/10 text-[9px] font-mono font-bold tracking-wide">
            <span className="text-red">DEFECTS: {cam1Defects.length}</span>
            <span className="text-white/20">|</span>
            <span className="text-green">QUALITY: {quality1}%</span>
          </div>
        )}

        {/* Video stream container */}
        <div className="aspect-[4/3] w-full bg-slate-950 flex items-center justify-center overflow-hidden relative">
          
          {/* Subtle Corner Markers */}
          <div className="absolute top-2 left-2 w-3 h-3 border-t-2 border-l-2 border-white/20 pointer-events-none z-10"></div>
          <div className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-white/20 pointer-events-none z-10"></div>
          <div className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-white/20 pointer-events-none z-10"></div>
          <div className="absolute bottom-2 right-2 w-3 h-3 border-b-2 border-r-2 border-white/20 pointer-events-none z-10"></div>
          
          {cam1Settings.enabled ? (
            isDemoMode ? (
              <canvas
                ref={canvasRef1}
                width={640}
                height={480}
                style={getFilterStyle(cam1Settings)}
                className="w-full h-full object-cover"
              />
            ) : (
              <img
                src="/video_feed/1"
                alt="MindVision Line Scan Stream"
                style={getFilterStyle(cam1Settings)}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            )
          ) : (
            <div className="flex flex-col items-center gap-2 text-gray text-xs font-mono">
              <ShieldAlert className="w-8 h-8 opacity-45 text-red" />
              <span>FEED OFFLINE</span>
            </div>
          )}

          {/* SVG Bounding Boxes Overlay */}
          {cam1Settings.enabled && (
            <svg 
              className="absolute inset-0 w-full h-full pointer-events-auto"
              viewBox="0 0 640 480"
              preserveAspectRatio="none"
            >
              {cam1Defects.map((def) => {
                const color = DEFECT_COLORS[def.type] || { border: '#E63946', fill: 'rgba(230,57,70,0.15)', text: '#ffffff', label: 'Defect' };
                return (
                  <g 
                    key={def.id} 
                    className="cursor-pointer group/box"
                    onClick={() => onSelectDefect(def)}
                  >
                    {/* Bounding box with glow */}
                    <rect
                      x={def.bbox.x}
                      y={def.bbox.y}
                      width={def.bbox.width}
                      height={def.bbox.height}
                      fill={color.fill}
                      stroke={color.border}
                      strokeWidth="2.5"
                      style={{ filter: `drop-shadow(0 0 4px ${color.border})` }}
                      className="transition-all duration-150 hover:stroke-white hover:stroke-[3]"
                    />
                    
                    {/* Label background */}
                    <rect
                      x={def.bbox.x}
                      y={def.bbox.y - 18}
                      width={def.bbox.width > 100 ? def.bbox.width : 100}
                      height={18}
                      fill={color.border}
                    />

                    {/* Label text */}
                    <text
                      x={def.bbox.x + 4}
                      y={def.bbox.y - 5}
                      fill={color.text}
                      fontSize="9"
                      fontWeight="bold"
                      fontFamily="IBM Plex Mono, monospace"
                    >
                      {color.label.toUpperCase()} {(def.confidence * 100).toFixed(0)}%
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        {/* Timestamp & Feed Specs Overlay */}
        {cam1Settings.enabled && (
          <div className="absolute bottom-3 left-3 right-3 z-10 flex justify-between items-center text-[9px] font-mono text-gray font-semibold">
            <span>RESOLUTION: 4096×200 STITCHED @ 30FPS</span>
            <span>TIME: {new Date().toLocaleTimeString()}</span>
          </div>
        )}
      </div>

    </div>
  );
}

export default CameraFeed;
