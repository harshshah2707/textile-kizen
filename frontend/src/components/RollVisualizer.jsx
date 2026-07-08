import React from 'react';

function RollVisualizer({ defectHistory, processedMeters, selectedDefect, onSelectDefect }) {
  // Let's set a target height for our visualizer track (e.g. 240px)
  const trackHeight = 280;
  
  return (
    <div className="glass-panel p-4 rounded-2xl border border-white/5 shadow-md flex flex-col gap-3">
      <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-cyan shadow-sm"></span>
        <span>Fabric Roll Defect Map</span>
      </h3>
      
      <div className="flex items-center gap-4 py-2">
        {/* Roll Cylinder Graphic */}
        <div className="flex flex-col items-center gap-1.5 select-none shrink-0">
          <div className="relative w-16 h-8 bg-gradient-to-r from-slate-700 via-slate-600 to-slate-800 rounded-full border border-white/10 flex items-center justify-center shadow-lg">
            <div className="absolute inset-0 bg-white/5 rounded-full blur-[1px]"></div>
            <div className="w-12 h-1 bg-white/25 rounded-full"></div>
          </div>
          <span className="text-[8px] font-mono text-gray uppercase font-bold">Loom Roll</span>
        </div>
        
        {/* Inspection Stats */}
        <div className="flex-1 flex flex-col gap-0.5 text-xs font-mono">
          <div className="text-[9px] text-gray uppercase font-bold">Current Scanned Length</div>
          <div className="text-white font-extrabold text-sm">{processedMeters.toFixed(2)} meters</div>
        </div>
      </div>

      {/* Fabric Ribbon Mapping Track */}
      <div className="relative w-full bg-navy/40 rounded-xl border border-white/5 overflow-hidden p-2 flex" style={{ height: `${trackHeight}px` }}>
        {/* Vertical Ruler */}
        <div className="w-10 h-full border-r border-white/5 flex flex-col justify-between text-[8px] font-mono text-gray shrink-0 pr-1.5">
          <span>0.0 m</span>
          <span>{(processedMeters / 2).toFixed(1)} m</span>
          <span>{processedMeters.toFixed(1)} m</span>
        </div>
        
        {/* Fabric Strip Area */}
        <div className="relative flex-1 h-full bg-slate-950/45 rounded-lg border border-white/5 overflow-hidden">
          {/* Weave grid background */}
          <div className="absolute inset-0 opacity-[0.03] bg-[radial-gradient(#ffffff_1px,transparent_1px)] [background-size:8px_8px] pointer-events-none"></div>
          
          {/* Scanned fabric flow bar */}
          <div className="absolute left-0 right-0 top-0 bg-blue-600/5 border-b border-blue-500/25 h-full transition-all duration-300"></div>
          
          {/* Defect Markers */}
          {defectHistory.map((d) => {
            const size = d.size_mm || 10;
            // Map X (0 to 4096) to left percentage
            const x_pct = d.bbox ? ((d.bbox.x + d.bbox.width/2) / 4096) * 100 : 50;
            
            // Map Y (distance_meters) to top percentage
            const distM = d.distance_meters !== undefined ? d.distance_meters : 0.0;
            const totalMeters = Math.max(1.0, processedMeters);
            const y_pct = (distM / totalMeters) * 100;
            
            const isSelected = selectedDefect && selectedDefect.id === d.id;
            
            // Defect color
            let markerColor = 'bg-red-500 shadow-sm';
            if (d.type === 'stain' || d.type === 'Needle mark') {
              markerColor = 'bg-orange-500 shadow-sm';
            } else if (d.type === 'horizontal' || d.type === 'lines') {
              markerColor = 'bg-yellow-500 shadow-sm';
            }
            
            return (
              <button
                key={d.id}
                onClick={() => onSelectDefect(d)}
                title={`D-${d.id} | ${d.type ? d.type.toUpperCase() : 'DEFECT'} | ${distM.toFixed(2)} m`}
                className={`absolute w-3.5 h-3.5 -ml-1.75 -mt-1.75 rounded-full border-2 border-white/90 cursor-pointer transition-all duration-200 hover:scale-130 z-10 ${markerColor} ${
                  isSelected ? 'scale-135 ring-4 ring-cyan border-cyan' : ''
                }`}
                style={{
                  left: `${x_pct}%`,
                  top: `${y_pct}%`
                }}
              />
            );
          })}
          
          {/* No defects label */}
          {defectHistory.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-[10px] text-gray uppercase tracking-widest font-mono select-none">
              No Defects Logged
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RollVisualizer;
