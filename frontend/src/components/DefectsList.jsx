import React, { useState, useEffect, useRef } from 'react';
import { 
  ChevronRight, 
  ChevronDown, 
  List, 
  Camera, 
  FileCheck,
  ArrowUpDown
} from 'lucide-react';

const TYPE_BADGE_CLASSES = {
  'Broken stitch': 'bg-rose-50 text-rose-700 border border-rose-200',
  'hole': 'bg-orange-50 text-orange-700 border border-orange-200',
  'horizontal': 'bg-amber-50 text-amber-700 border border-amber-200',
  'lines': 'bg-purple-50 text-purple-700 border border-purple-200',
  'Needle mark': 'bg-emerald-50 text-emerald-700 border border-emerald-200',
  'Pinched fabric': 'bg-blue-50 text-blue-700 border border-blue-200',
  'stain': 'bg-pink-50 text-pink-700 border border-pink-200',
  'Vertical': 'bg-cyan-50 text-cyan-700 border border-cyan-200',
  'defect': 'bg-slate-50 text-slate-700 border border-slate-200'
};

const TYPE_BORDER_CLASSES = {
  'Broken stitch': 'border-l-rose-500',
  'hole': 'border-l-orange-500',
  'horizontal': 'border-l-yellow-500',
  'lines': 'border-l-purple-500',
  'Needle mark': 'border-l-emerald-500',
  'Pinched fabric': 'border-l-blue-500',
  'stain': 'border-l-pink-500',
  'Vertical': 'border-l-cyan',
  'defect': 'border-l-slate-500'
};

// Canvas-based component to render a mock zoomed-in crop of the fabric defect
function DefectCropCanvas({ defect }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    // Draw Zoomed Fabric base
    ctx.fillStyle = '#E6ECEF';
    ctx.fillRect(0, 0, w, h);

    // Draw grid thread patterns (zoomed)
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.05)';
    ctx.lineWidth = 1.5;
    for (let x = 0; x < w; x += 12) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 12) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Draw zoom focal guides
    ctx.strokeStyle = 'rgba(37, 99, 235, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(w/2, h/2, w/3, 0, 2*Math.PI);
    ctx.stroke();

    // Draw defect shape zoomed at center
    ctx.save();
    if (defect.type === 'Broken stitch') {
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 3.5;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(w/2 - 40, h/2 - 10);
      ctx.lineTo(w/2 + 40, h/2 + 10);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = '#ff3e00';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(w/2 - 10, h/2 - 3, 4, 0, 2*Math.PI);
      ctx.stroke();
    } else if (defect.type === 'hole') {
      ctx.fillStyle = '#050505';
      ctx.strokeStyle = 'rgba(255, 87, 34, 0.7)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(w/2, h/2, 22, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.strokeStyle = '#444';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w/2 - 22, h/2); ctx.lineTo(w/2 + 22, h/2);
      ctx.moveTo(w/2, h/2 - 22); ctx.lineTo(w/2, h/2 + 22);
      ctx.stroke();
    } else if (defect.type === 'horizontal') {
      ctx.strokeStyle = 'rgba(234, 179, 8, 0.8)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(w/2 - 45, h/2);
      ctx.lineTo(w/2 + 45, h/2);
      ctx.stroke();
    } else if (defect.type === 'lines') {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(w/2 - 30, h/2 - 15); ctx.lineTo(w/2 + 30, h/2 - 15);
      ctx.moveTo(w/2 - 35, h/2);      ctx.lineTo(w/2 + 35, h/2);
      ctx.moveTo(w/2 - 30, h/2 + 15); ctx.lineTo(w/2 + 30, h/2 + 15);
      ctx.stroke();
    } else if (defect.type === 'Needle mark') {
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(w/2 - 15, h/2 - 10, 3, 0, 2*Math.PI);
      ctx.arc(w/2, h/2, 3, 0, 2*Math.PI);
      ctx.arc(w/2 + 15, h/2 + 10, 3, 0, 2*Math.PI);
      ctx.fill();
    } else if (defect.type === 'Pinched fabric') {
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(w/2 - 40, h/2 + 15);
      ctx.bezierCurveTo(w/2 - 20, h/2 - 15, w/2 + 20, h/2 - 15, w/2 + 40, h/2 + 15);
      ctx.stroke();
    } else if (defect.type === 'stain') {
      const grad = ctx.createRadialGradient(w/2, h/2, 2, w/2, h/2, 30);
      grad.addColorStop(0, 'rgba(236, 72, 153, 0.5)');
      grad.addColorStop(0.8, 'rgba(236, 72, 153, 0.15)');
      grad.addColorStop(1, 'rgba(236, 72, 153, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(w/2, h/2, 30, 0, 2 * Math.PI);
      ctx.fill();
    } else if (defect.type === 'Vertical') {
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(w/2, h/2 - 45);
      ctx.lineTo(w/2, h/2 + 45);
      ctx.stroke();
    } else {
      ctx.strokeStyle = 'rgba(230, 57, 70, 0.8)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(w/2, h/2, 15, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.restore();

    // Draw box overlay
    ctx.strokeStyle = '#00B8D4';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(w/2 - 35, h/2 - 35, 70, 70);

    // Bounding Box text
    ctx.fillStyle = '#00B8D4';
    ctx.font = 'bold 9px "IBM Plex Mono", monospace';
    ctx.fillText('ROI CROP', w/2 - 30, h/2 - 40);

  }, [defect]);

  return (
    <canvas
      ref={canvasRef}
      width={120}
      height={120}
      className="rounded-xl border border-slate-200 shadow-inner shadow-slate-300 object-cover"
    />
  );
}

function DefectsList({ defectHistory, selectedDefect, onSelectDefect, calculateDefectPoints }) {
  const [expandedId, setExpandedId] = useState(null);
  const [sortField, setSortField] = useState('timestamp'); // timestamp, type, confidence, size_mm, points, camera
  const [sortAsc, setSortAsc] = useState(false);
  const [savedDefectIds, setSavedDefectIds] = useState(new Set());

  // Automatically expand a row if the defect is clicked/selected on camera canvas
  useEffect(() => {
    if (selectedDefect) {
      setExpandedId(selectedDefect.id);
      const elem = document.getElementById(`defect-row-${selectedDefect.id}`);
      if (elem) {
        elem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [selectedDefect]);

  const toggleRow = (id, def) => {
    if (expandedId === id) {
      setExpandedId(null);
      if (selectedDefect && selectedDefect.id === id) {
        onSelectDefect(null);
      }
    } else {
      setExpandedId(id);
      onSelectDefect(def);
    }
  };

  const toggleSaveDefect = (id, e) => {
    e.stopPropagation();
    setSavedDefectIds(prev => {
      const copy = new Set(prev);
      if (copy.has(id)) {
        copy.delete(id);
      } else {
        copy.add(id);
      }
      return copy;
    });
  };

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  // Sort and Slice history to last 20
  const getSortedDefects = () => {
    const list = defectHistory.map(d => ({
      ...d,
      points: calculateDefectPoints ? calculateDefectPoints(d.type, d.size_mm) : 1
    }));
    
    list.sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      
      if (sortField === 'type') {
        valA = a.type.toLowerCase();
        valB = b.type.toLowerCase();
      }

      if (valA < valB) return sortAsc ? -1 : 1;
      if (valA > valB) return sortAsc ? 1 : -1;
      return 0;
    });
    return list.slice(0, 20);
  };

  const sortedDefects = getSortedDefects();

  return (
    <div className="glass-panel rounded-2xl border border-white/5 shadow-lg flex-1 flex flex-col min-h-[350px] overflow-hidden">
      
      {/* Card Header */}
      <div className="px-5 py-4 border-b border-white/5 flex justify-between items-center bg-white/[0.01]">
        <div className="flex items-center gap-2">
          <List className="w-4 h-4 text-cyan" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-white font-sans">Recent Defects Log</h3>
        </div>
        <span className="text-[10px] bg-white/5 border border-white/10 rounded-full px-2 py-0.5 font-bold font-mono text-gray">
          Displaying last 20 records
        </span>
      </div>

      {/* Table Container */}
      <div className="flex-1 overflow-y-auto max-h-[360px] relative">
        {defectHistory.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray text-xs gap-2 py-12 font-mono">
            <Camera className="w-8 h-8 opacity-30 text-cyan animate-pulse" />
            <span>SCANNING FABRIC MATRIX... NO DEFECTS DETECTED.</span>
          </div>
        ) : (
          <table className="w-full text-left border-collapse text-[11px] text-gray">
            <thead>
              <tr className="border-b border-white/5 bg-navy/60 text-white/90 select-none sticky top-0 z-10 backdrop-blur-md">
                <th className="w-8 py-2.5 px-3"></th>
                <th className="py-2.5 px-3 cursor-pointer hover:text-cyan transition-colors" onClick={() => handleSort('id')}>
                  <div className="flex items-center gap-1 font-mono">
                    <span>ID</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
                <th className="py-2.5 px-3 cursor-pointer hover:text-cyan transition-colors" onClick={() => handleSort('type')}>
                  <div className="flex items-center gap-1 font-sans">
                    <span>Type</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
                <th className="py-2.5 px-3 cursor-pointer hover:text-cyan transition-colors" onClick={() => handleSort('confidence')}>
                  <div className="flex items-center gap-1 font-sans">
                    <span>Conf.</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
                <th className="py-2.5 px-3 cursor-pointer hover:text-cyan transition-colors" onClick={() => handleSort('size_mm')}>
                  <div className="flex items-center gap-1 font-sans">
                    <span>Size</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
                <th className="py-2.5 px-3 cursor-pointer hover:text-cyan transition-colors" onClick={() => handleSort('points')}>
                  <div className="flex items-center gap-1 font-sans">
                    <span>Points</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
                <th className="py-2.5 px-3 cursor-pointer hover:text-cyan transition-colors" onClick={() => handleSort('camera')}>
                  <div className="flex items-center gap-1 font-sans">
                    <span>Camera</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
                <th className="py-2.5 px-3 cursor-pointer hover:text-cyan transition-colors" onClick={() => handleSort('timestamp')}>
                  <div className="flex items-center gap-1 font-sans">
                    <span>Timestamp</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedDefects.map((def) => {
                const isExpanded = expandedId === def.id;
                const isSelected = selectedDefect && selectedDefect.id === def.id;
                const isSaved = savedDefectIds.has(def.id);
                
                return (
                  <React.Fragment key={def.id}>
                    {/* Main Row */}
                    <tr
                      id={`defect-row-${def.id}`}
                      onClick={() => toggleRow(def.id, def)}
                      className={`border-b border-white/5 border-l-2 cursor-pointer transition-all hover:bg-white/[0.02] select-none ${
                        isSelected 
                          ? 'bg-cyan/5 text-white border-l-cyan font-bold' 
                          : `${TYPE_BORDER_CLASSES[def.type] || 'border-l-transparent'}`
                      }`}
                    >
                      <td className="py-2.5 px-3 text-center">
                        {isExpanded ? (
                          <ChevronDown className="w-3.5 h-3.5 text-cyan" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5 text-gray hover:text-white" />
                        )}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-white font-medium">{def.id}</td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${TYPE_BADGE_CLASSES[def.type] || 'bg-slate-500/10 text-slate-400 border border-slate-500/20'}`}>
                          {def.type}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono font-bold text-white">{(def.confidence * 100).toFixed(0)}%</td>
                      <td className="py-2.5 px-3 font-mono text-white">{def.size_mm} mm</td>
                      <td className="py-2.5 px-3 font-mono font-extrabold text-orange">
                        {def.points} pt{def.points > 1 ? 's' : ''}
                      </td>
                      <td className="py-2.5 px-3 font-mono">CAM 0{def.camera}</td>
                      <td className="py-2.5 px-3 font-mono text-gray">
                        {new Date(def.timestamp).toLocaleTimeString()}
                      </td>
                    </tr>

                    {/* Expandable Details Row */}
                    {isExpanded && (
                      <tr className="bg-navy/35 border-b border-white/5">
                        <td colSpan="8" className="p-4">
                          <div className="flex flex-col md:flex-row gap-4 items-start">
                            
                            {/* Zooms crop visual */}
                            <DefectCropCanvas defect={def} />

                            {/* Metadata list */}
                            <div className="flex-1 grid grid-cols-2 gap-x-6 gap-y-2 text-[10px] font-mono">
                              <div>
                                <span className="text-gray font-sans uppercase">Unique Identifier:</span>
                                <p className="text-white font-semibold">{def.id}</p>
                              </div>
                              <div>
                                <span className="text-gray font-sans uppercase">Coordinates (X,Y):</span>
                                <p className="text-white font-semibold">{def.location}</p>
                              </div>
                              <div>
                                <span className="text-gray font-sans uppercase">YOLO Box Area:</span>
                                <p className="text-white font-semibold">
                                  {def.bbox.width}px x {def.bbox.height}px
                                </p>
                              </div>
                              <div>
                                <span className="text-gray font-sans uppercase">Estimated Size:</span>
                                <p className="text-white font-semibold">{def.size_mm} mm</p>
                              </div>
                              <div>
                                <span className="text-gray font-sans uppercase">ASTM Penalty Points:</span>
                                <p className="text-orange font-bold text-[11px]">
                                  {def.points} Point{def.points > 1 ? 's' : ''} (ASTM D5430)
                                </p>
                              </div>
                              <div>
                                <span className="text-gray font-sans uppercase">Analysis Epoch:</span>
                                <p className="text-white font-semibold">
                                  {new Date(def.timestamp).toISOString().replace('T', ' ').substring(0, 19)}
                                </p>
                              </div>
                              <div>
                                <span className="text-gray font-sans uppercase">Defect Status:</span>
                                <p className={`font-sans font-semibold uppercase ${isSaved ? 'text-green' : 'text-orange'}`}>
                                  {isSaved ? '✓ Saved for review' : '⚠ Action Pending'}
                                </p>
                              </div>
                            </div>

                            {/* Right action block */}
                            <div className="flex flex-col gap-2 w-full md:w-32 justify-end h-full">
                              <button
                                onClick={(e) => toggleSaveDefect(def.id, e)}
                                className={`py-1.5 px-3 rounded-lg text-[9px] font-bold uppercase tracking-wider flex items-center justify-center gap-1 shadow-sm transition-all border cursor-pointer ${
                                  isSaved 
                                    ? 'bg-green/10 border-green/30 text-green shadow-sm' 
                                    : 'bg-white/5 border-white/10 hover:border-white/20 text-white'
                                }`}
                              >
                                <FileCheck className="w-3.5 h-3.5" />
                                <span>{isSaved ? 'Saved' : 'Save Item'}</span>
                              </button>
                            </div>

                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

    </div>
  );
}

export default DefectsList;
