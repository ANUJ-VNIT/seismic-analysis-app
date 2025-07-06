import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend #changed

def create_sdof_frame_animation(time, displacement, velocity=None, acceleration=None, #changed
                              save_animation=False, filename='sdof_frame.gif'): #changed
    
    displacement_um = displacement * 1e6
    
    if velocity is None: #changed
        velocity = np.gradient(displacement, time) #changed
    if acceleration is None: #changed
        acceleration = np.gradient(velocity, time) #changed
    
    fig = plt.figure(figsize=(14, 8))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    ax1.set_title('Single DOF Frame Deformation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_xlim(0, time[-1])
    ax2.set_ylim(min(displacement_um)*1.1, max(displacement_um)*1.1)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Displacement (μm)', fontsize=12)
    ax2.set_title('Displacement vs Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Frame elements
    base_line, = ax1.plot([0, 30], [0, 0], 'k-', linewidth=6)
    col_left_line, = ax1.plot([], [], 'steelblue', linewidth=5)
    col_right_line, = ax1.plot([], [], 'steelblue', linewidth=5)
    beam_line, = ax1.plot([], [], 'steelblue', linewidth=6)
    trace_line, = ax2.plot([], [], 'r-', linewidth=2)
    current_point, = ax2.plot([], [], 'ro', markersize=8)
    
    # Text elements
    text_time = ax1.text(-45, 95, '', fontsize=11, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.7))
    text_disp = ax1.text(-45, 88, '', fontsize=11, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.7))
    text_vel = ax1.text(-45, 81, '', fontsize=11, fontweight='bold', 
                       bbox=dict(facecolor='white', alpha=0.7))
    text_acc = ax1.text(-45, 74, '', fontsize=11, fontweight='bold', 
                       bbox=dict(facecolor='white', alpha=0.7))
    
    def init():
        col_left_line.set_data([], [])
        col_right_line.set_data([], [])
        beam_line.set_data([], [])
        trace_line.set_data([], [])
        current_point.set_data([], [])
        return col_left_line, col_right_line, beam_line, trace_line, current_point
    
    def animate(i):
        if i >= len(time): #changed
            return init() #changed
        
        t = time[i]
        u = displacement[i]
        u_um = displacement_um[i]
        v = velocity[i]
        a = acceleration[i]
        
        scale = 1000
        dx = scale * u
        
        col_left_line.set_data([0, 0 + dx], [0, 80])
        col_right_line.set_data([30, 30 + dx], [0, 80])
        beam_line.set_data([0 + dx, 30 + dx], [80, 80])
        trace_line.set_data(time[:i+1], displacement_um[:i+1])
        current_point.set_data([t], [u_um])
        
        text_time.set_text(f'Time: {t:.4f} s')
        text_disp.set_text(f'Disp: {u_um:.3f} μm')
        text_vel.set_text(f'Vel: {v:.6f} m/s')
        text_acc.set_text(f'Acc: {a:.6f} m/s²')
        
        return col_left_line, col_right_line, beam_line, trace_line, current_point
    
    # Calculate appropriate fps (limit between 5-30 fps) #changed
    duration = time[-1] #changed
    target_fps = min(max(len(time) / duration, 5), 30) #changed
    interval = 1000 / target_fps #changed
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(time), interval=interval, #changed
        blit=False, repeat=False
    )
    
    plt.tight_layout()
    
    # Create GIF buffer
    gif_buffer = BytesIO()
    try: #changed
        # Save to buffer with optimized settings #changed
        anim.save(gif_buffer, format='gif', writer='pillow', 
                 fps=target_fps, dpi=80, bitrate=1800) #changed
        gif_buffer.seek(0) #changed
        
        # Optionally save file #changed
        if save_animation: #changed
            anim.save(filename, writer='pillow', fps=target_fps, dpi=100) #changed
            print(f"✅ Animation saved as {filename}") #changed
            
    except Exception as e: #changed
        print(f"Error creating animation: {e}") #changed
        return None #changed
    finally: #changed
        plt.close(fig) #changed
    
    return gif_buffer