import React from 'react';

function KizenLogo({ className = "h-16 w-auto", lightText = true, gearOnly = false }) {
  if (gearOnly) {
    return (
      <img 
        src="/kizen_gear.png" 
        className={className} 
        alt="Kizen Gear" 
        style={{ objectFit: 'contain' }}
        onError={(e) => {
          e.target.style.display = 'none';
        }}
      />
    );
  }

  return (
    <img 
      src="/kizen_logo_light.png" 
      className={className} 
      alt="Kizen Engineering" 
      style={{ objectFit: 'contain' }}
      onError={(e) => {
        e.target.style.display = 'none';
      }}
    />
  );
}

export default KizenLogo;
