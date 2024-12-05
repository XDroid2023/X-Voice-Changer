import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.signal import hilbert
import time

class VoiceChanger:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("X Voice Changer 🎤")
        self.root.geometry("800x700")
        
        self.sample_rate = 44100
        self.recording = False
        self.audio_data = None
        self.record_start_time = None
        
        self.effect_params = {
            'pitch': 1.0,
            'reverb': 0.0,
            'distortion': 0.0,
            'delay': 0.0,
            'flanger': 0.0
        }
        
        # Create main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="X Voice Changer",
            font=('Arial', 24, 'bold')
        )
        title_label.pack(pady=20)
        
        # Recording indicator
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(pady=10)
        
        self.record_indicator = tk.Canvas(
            self.status_frame,
            width=20,
            height=20,
            bg='gray'
        )
        self.record_indicator.pack(side=tk.LEFT, padx=5)
        
        self.time_label = ttk.Label(
            self.status_frame,
            text="00:00",
            font=('Arial', 14)
        )
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready to record",
            font=('Arial', 12)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Control buttons
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(pady=20)
        
        ttk.Button(
            control_frame,
            text="▶ START",
            command=self.start_recording
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="⏹ STOP",
            command=self.stop_recording
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="🔊 PLAY",
            command=self.play_audio
        ).pack(side=tk.LEFT, padx=5)
        
        # Effect parameters
        params_frame = ttk.LabelFrame(self.main_frame, text="Effect Controls", padding=15)
        params_frame.pack(fill=tk.X, pady=20)
        
        # Create sliders
        sliders = [
            ("Pitch", 'pitch', 0.5, 2.0),
            ("Reverb", 'reverb', 0.0, 1.0),
            ("Distortion", 'distortion', 0.0, 1.0),
            ("Delay", 'delay', 0.0, 1.0),
            ("Flanger", 'flanger', 0.0, 1.0)
        ]
        
        for text, param, min_val, max_val in sliders:
            frame = ttk.Frame(params_frame)
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=text).pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                orient=tk.HORIZONTAL,
                value=self.effect_params[param],
                command=lambda v, p=param: self.update_effect_param(p, float(v))
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Effect buttons
        effects_frame = ttk.LabelFrame(self.main_frame, text="Voice Effects", padding=15)
        effects_frame.pack(fill=tk.X, pady=20)
        
        effects = [
            ("NORMAL", "normal"),
            ("ALIEN", "alien"),
            ("ROBOT", "robot"),
            ("CHIPMUNK", "chipmunk"),
            ("DEMON", "demon"),
            ("UNDERWATER", "underwater"),
            ("RADIO", "radio"),
            ("CHORUS", "chorus"),
            ("CYBERPUNK", "cyberpunk"),
            ("CAVE", "cave"),
            ("8-BIT", "8bit"),
            ("TELEPHONE", "telephone"),
            ("STADIUM", "stadium"),
            ("VINYL", "vinyl"),
            ("VOCODER", "vocoder"),
            ("SPACE", "space"),
            ("DRAGON", "dragon"),
            ("CRYSTAL", "crystal"),
            ("STORM", "storm"),
            ("TIME-STRETCH", "timestretch"),
            ("REVERSE-ECHO", "reverseecho"),
            ("METALLIC", "metallic"),
            ("GHOSTLY", "ghostly"),
            ("QUANTUM", "quantum")
        ]
        
        self.effect_var = tk.StringVar(value="normal")
        
        # Create effect buttons in a grid
        button_frame = ttk.Frame(effects_frame)
        button_frame.pack(fill=tk.X)
        
        row = 0
        col = 0
        for text, value in effects:
            ttk.Button(
                button_frame,
                text=text,
                command=lambda v=value: self.effect_var.set(v)
            ).grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            
            col += 1
            if col > 3:  # 4 buttons per row
                col = 0
                row += 1
        
        # Configure grid columns to be equal width
        for i in range(4):
            button_frame.grid_columnconfigure(i, weight=1)

    def update_effect_param(self, param, value):
        self.effect_params[param] = value
        
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.record_start_time = time.time()
        self.update_record_indicator()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.extend(indata[:, 0])
        
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=audio_callback
        )
        self.stream.start()
        
    def stop_recording(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.recording = False
        self.record_start_time = None
        self.record_indicator.configure(bg='gray')
        self.time_label.configure(text="00:00")
        self.status_label.configure(text="Recording stopped")
        
    def update_record_indicator(self):
        if self.recording:
            # Toggle indicator color
            current_color = self.record_indicator.cget('bg')
            new_color = 'gray' if current_color == 'red' else 'red'
            self.record_indicator.configure(bg=new_color)
            
            # Update recording time
            if self.record_start_time:
                elapsed = int(time.time() - self.record_start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                self.time_label.configure(text=f"{minutes:02d}:{seconds:02d}")
                self.status_label.configure(text="Recording in progress...")
            
            # Schedule next update
            self.root.after(500, self.update_record_indicator)
        
    def play_audio(self):
        if not hasattr(self, 'audio_data') or not self.audio_data:
            return
            
        modified_audio = self.apply_effect(self.audio_data)
        sd.play(modified_audio, self.sample_rate)
        sd.wait()

    def apply_effect(self, audio_data):
        effect = self.effect_var.get()
        modified_audio = np.array(audio_data)
        
        # Apply pitch shift
        pitch_factor = self.effect_params['pitch']
        modified_audio = signal.resample(modified_audio, int(len(modified_audio) / pitch_factor))
        
        # Apply effects based on selection
        if effect == "alien":
            # Alien effect: Frequency modulation with phase shift
            t = np.arange(len(modified_audio)) / self.sample_rate
            carrier1 = np.sin(2 * np.pi * 50 * t + np.sin(2 * np.pi * 2 * t))
            carrier2 = np.cos(2 * np.pi * 30 * t + np.sin(2 * np.pi * 1.5 * t))
            modified_audio = modified_audio * (carrier1 + carrier2) * 0.5
            
        elif effect == "robot":
            # Enhanced robot effect: Ring modulation + bit crushing
            t = np.arange(len(modified_audio)) / self.sample_rate
            carrier = np.sign(np.sin(2 * np.pi * 50 * t))
            modified_audio = modified_audio * carrier
            # Simulate bit crushing
            bits = 6
            modified_audio = np.round(modified_audio * 2**(bits-1)) / 2**(bits-1)
            
        elif effect == "demon":
            # Enhanced demon effect: Pitch down + distortion + subharmonics
            modified_audio = signal.resample(modified_audio, int(len(modified_audio) * 1.5))
            modified_audio = np.clip(modified_audio * 2.0, -1, 1)
            # Add subharmonics
            t = np.arange(len(modified_audio)) / self.sample_rate
            sub = np.sin(2 * np.pi * 30 * t) * modified_audio * 0.3
            modified_audio += sub
            
        elif effect == "underwater":
            # Enhanced underwater effect: Low-pass filter + modulated reverb
            b, a = signal.butter(4, 0.1, 'low')
            modified_audio = signal.filtfilt(b, a, modified_audio)
            # Add modulated reverb
            t = np.arange(len(modified_audio)) / self.sample_rate
            mod = np.sin(2 * np.pi * 0.5 * t)
            delay_samples = int(0.1 * self.sample_rate)
            reverb = np.zeros_like(modified_audio)
            reverb[delay_samples:] = modified_audio[:-delay_samples] * 0.6 * (1 + 0.2 * mod[:-delay_samples])
            modified_audio += reverb
            
        elif effect == "radio":
            # Enhanced radio effect: Band-pass filter + noise + amplitude modulation
            b, a = signal.butter(4, [0.1, 0.3], 'band')
            modified_audio = signal.filtfilt(b, a, modified_audio)
            # Add noise
            noise = np.random.normal(0, 0.05, len(modified_audio))
            # Add amplitude modulation
            t = np.arange(len(modified_audio)) / self.sample_rate
            am = 1 + 0.2 * np.sin(2 * np.pi * 8 * t)
            modified_audio = (modified_audio + noise) * am
            
        elif effect == "chorus":
            # Enhanced chorus effect: Multiple modulated delays
            delays = [0.02, 0.03, 0.04]
            rates = [0.5, 1.0, 1.5]
            depths = [0.001, 0.002, 0.003]
            chorus = np.zeros_like(modified_audio)
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            for delay, rate, depth in zip(delays, rates, depths):
                mod_delay = delay + depth * np.sin(2 * np.pi * rate * t)
                delay_samples = (mod_delay * self.sample_rate).astype(int)
                delayed = np.zeros_like(modified_audio)
                for i, d in enumerate(delay_samples):
                    if i + d < len(modified_audio):
                        delayed[i] = modified_audio[i + d] * 0.3
                chorus += delayed
            modified_audio = modified_audio + chorus
            
        elif effect == "cyberpunk":
            # Cyberpunk effect: Glitch + distortion + frequency shifting
            t = np.arange(len(modified_audio)) / self.sample_rate
            # Add glitch effects
            glitch_env = np.ones_like(modified_audio)
            for _ in range(5):
                pos = np.random.randint(0, len(modified_audio))
                width = np.random.randint(100, 5000)
                if pos + width < len(modified_audio):
                    glitch_env[pos:pos+width] = np.random.choice([-1, 0, 1])
            # Add frequency shifting
            carrier = np.exp(2j * np.pi * 100 * t)
            modified_audio = np.real(hilbert(modified_audio) * carrier) * glitch_env
            
        elif effect == "cave":
            # Cave effect: Long reverb with frequency-dependent decay
            reverb_times = [0.1, 0.2, 0.3, 0.4, 0.5]
            reverb = np.zeros_like(modified_audio)
            for delay_time in reverb_times:
                delay_samples = int(delay_time * self.sample_rate)
                decay = np.exp(-3 * delay_time)
                delayed = np.zeros_like(modified_audio)
                delayed[delay_samples:] = modified_audio[:-delay_samples] * decay
                reverb += delayed
            modified_audio = modified_audio + reverb * 0.6
            
        elif effect == "8bit":
            # 8-bit gaming console effect
            bits = 4
            rate_reduce = 4
            # Reduce bit depth
            modified_audio = np.round(modified_audio * 2**(bits-1)) / 2**(bits-1)
            # Reduce sample rate
            modified_audio = modified_audio[::rate_reduce]
            modified_audio = np.repeat(modified_audio, rate_reduce)
            # Add square wave modulation
            t = np.arange(len(modified_audio)) / self.sample_rate
            square = np.sign(np.sin(2 * np.pi * 440 * t))
            modified_audio = modified_audio * (1 + 0.2 * square)
            
        elif effect == "telephone":
            # Old telephone effect: Band-pass filter + distortion + noise
            b, a = signal.butter(4, [0.2, 0.3], 'band')
            modified_audio = signal.filtfilt(b, a, modified_audio)
            # Add noise
            noise = np.random.normal(0, 0.02, len(modified_audio))
            # Add periodic interference
            t = np.arange(len(modified_audio)) / self.sample_rate
            interference = 0.1 * np.sin(2 * np.pi * 50 * t)
            modified_audio = modified_audio + noise + interference
            
        elif effect == "stadium":
            # Stadium effect: Multiple echoes with spatial simulation
            echoes = []
            for delay in np.linspace(0.1, 0.5, 5):
                delay_samples = int(delay * self.sample_rate)
                echo = np.zeros_like(modified_audio)
                echo[delay_samples:] = modified_audio[:-delay_samples] * (0.7 * np.exp(-delay))
                echoes.append(echo)
            modified_audio = modified_audio + sum(echoes) * 0.5
            
        elif effect == "vinyl":
            # Vinyl record effect: Crackle + wow/flutter
            # Add crackle noise
            crackle = np.random.normal(0, 0.02, len(modified_audio))
            crackle = np.where(np.abs(crackle) > 0.05, crackle, 0)
            # Add wow/flutter
            t = np.arange(len(modified_audio)) / self.sample_rate
            wow = 1 + 0.003 * np.sin(2 * np.pi * 0.5 * t)
            flutter = 1 + 0.001 * np.sin(2 * np.pi * 7 * t)
            modified_audio = modified_audio * wow * flutter + crackle
            
        elif effect == "vocoder":
            # Vocoder effect: Frequency band modulation
            num_bands = 6
            bands = []
            for i in range(num_bands):
                freq = 100 * (i + 1)
                t = np.arange(len(modified_audio)) / self.sample_rate
                carrier = np.sin(2 * np.pi * freq * t)
                # Extract envelope
                envelope = np.abs(hilbert(modified_audio))
                # Smooth envelope
                b, a = signal.butter(2, 10 / (self.sample_rate/2))
                envelope = signal.filtfilt(b, a, envelope)
                bands.append(carrier * envelope)
            modified_audio = sum(bands) / num_bands
            
        elif effect == "space":
            # Space effect: Frequency shifting + long reverb + modulation
            t = np.arange(len(modified_audio)) / self.sample_rate
            # Frequency shifting
            carrier = np.exp(2j * np.pi * 50 * t)
            modified_audio = np.real(hilbert(modified_audio) * carrier)
            # Long reverb
            reverb_times = np.linspace(0.1, 2.0, 8)
            reverb = np.zeros_like(modified_audio)
            for delay_time in reverb_times:
                delay_samples = int(delay_time * self.sample_rate)
                if delay_samples < len(modified_audio):
                    delayed = np.zeros_like(modified_audio)
                    delayed[delay_samples:] = modified_audio[:-delay_samples] * 0.3
                    mod = np.sin(2 * np.pi * 0.1 * t)
                    reverb += delayed * (1 + 0.2 * mod)
            modified_audio = modified_audio + reverb * 0.4
            
        elif effect == "dragon":
            # Dragon effect: Fire-breathing sound with growl
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            # Add growl using frequency modulation
            growl_freq = 30
            growl = 0.3 * np.sin(2 * np.pi * growl_freq * t)
            
            # Create crackling fire sound
            fire = np.random.normal(0, 0.1, len(modified_audio))
            fire = np.where(np.abs(fire) > 0.15, fire, 0)
            
            # Add low frequency rumble
            rumble = 0.2 * np.sin(2 * np.pi * 20 * t)
            
            # Pitch shift down and add distortion
            modified_audio = signal.resample(modified_audio, int(len(modified_audio) * 1.3))
            modified_audio = modified_audio[:len(t)]  # Ensure same length
            modified_audio = np.clip(modified_audio * 2.0, -1, 1)
            
            # Combine all effects
            modified_audio = modified_audio * (1 + growl) + fire + rumble
            modified_audio = np.clip(modified_audio, -1, 1)

        elif effect == "crystal":
            # Crystal effect: Clear, bell-like tones with shimmer
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            # Create bell-like harmonics
            harmonics = [1.0, 2.0, 3.0, 5.0, 8.0]
            bell = np.zeros_like(modified_audio)
            for h in harmonics:
                bell += 0.2 * np.sin(2 * np.pi * 800 * h * t) * np.exp(-3 * t)
            
            # Add shimmer effect
            shimmer = np.sin(2 * np.pi * 2000 * t) * np.sin(2 * np.pi * 4 * t)
            
            # Apply ring modulation
            carrier = 1 + 0.3 * (bell + 0.2 * shimmer)
            modified_audio = modified_audio * carrier
            
            # Add gentle chorus
            chorus = np.zeros_like(modified_audio)
            for delay in [0.01, 0.02]:
                delay_samples = int(delay * self.sample_rate)
                if delay_samples < len(modified_audio):
                    chorus[delay_samples:] += modified_audio[:-delay_samples] * 0.3
            
            modified_audio = modified_audio + chorus
            modified_audio = np.clip(modified_audio, -1, 1)

        elif effect == "storm":
            # Storm effect: Thunder and wind sounds
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            # Create thunder rumble
            thunder = np.random.normal(0, 0.2, len(modified_audio))
            thunder_env = np.exp(-2 * t)
            thunder = thunder * thunder_env
            
            # Create wind howl
            wind_freq = 2 * (1 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
            wind = 0.3 * np.sin(2 * np.pi * 100 * t * wind_freq)
            
            # Add rain effect
            rain = np.random.normal(0, 0.05, len(modified_audio))
            rain = np.where(np.abs(rain) > 0.08, rain, 0)
            
            # Combine with voice
            modified_audio = modified_audio + thunder + wind + rain
            modified_audio = np.clip(modified_audio, -1, 1)

        elif effect == "timestretch":
            # Time-stretch effect: Variable speed playback
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            # Create time-varying stretch factor
            stretch = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
            
            # Resample with variable rate
            new_audio = np.zeros_like(modified_audio)
            window = 1000  # Process in small windows
            for i in range(0, len(modified_audio), window):
                chunk = modified_audio[i:min(i+window, len(modified_audio))]
                stretch_factor = stretch[i]
                stretched = signal.resample(chunk, int(len(chunk) * stretch_factor))
                if i + len(stretched) <= len(new_audio):
                    new_audio[i:i+len(stretched)] = stretched
            
            modified_audio = new_audio

        elif effect == "reverseecho":
            # Reverse echo effect: Echoes appear before the sound
            chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
            modified = np.zeros_like(modified_audio)
            
            for i in range(0, len(modified_audio), chunk_size):
                chunk = modified_audio[i:min(i+chunk_size, len(modified_audio))]
                # Create reverse echo
                echo = np.flip(chunk)
                echo = echo * 0.3  # Reduce echo volume
                
                # Add echo before current chunk
                if i >= chunk_size:
                    modified[i-chunk_size:i] += echo[:chunk_size]
                modified[i:i+len(chunk)] += chunk
            
            modified_audio = modified

        elif effect == "metallic":
            # Metallic effect: Ring modulation with harmonic resonance
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            # Create metallic resonances
            resonances = [300, 600, 1200, 2400]
            metal = np.zeros_like(modified_audio)
            for freq in resonances:
                phase = np.random.random() * 2 * np.pi
                metal += np.sin(2 * np.pi * freq * t + phase)
            
            # Add ring modulation
            carrier = 1 + 0.3 * metal
            modified_audio = modified_audio * carrier
            
            # Add metallic reverb
            reverb = np.zeros_like(modified_audio)
            for delay in [0.02, 0.03, 0.04]:
                delay_samples = int(delay * self.sample_rate)
                if delay_samples < len(modified_audio):
                    reverb[delay_samples:] += modified_audio[:-delay_samples] * 0.2
            
            modified_audio = modified_audio + reverb
            modified_audio = np.clip(modified_audio, -1, 1)

        elif effect == "ghostly":
            # Ghostly effect: Spectral smearing and pitch variations
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            # Create spectral smearing
            smear = signal.convolve(modified_audio, np.hanning(1000), mode='same')
            
            # Add wandering pitch
            pitch_mod = 1 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
            
            # Create ethereal harmonics
            harmonics = np.zeros_like(modified_audio)
            for i in range(1, 4):
                phase = np.random.random() * 2 * np.pi
                harmonics += 0.1 * np.sin(2 * np.pi * 200 * i * t + phase)
            
            # Combine effects
            modified_audio = smear * pitch_mod + harmonics
            
            # Add long, ethereal reverb
            reverb = np.zeros_like(modified_audio)
            for delay in np.linspace(0.1, 1.0, 5):
                delay_samples = int(delay * self.sample_rate)
                if delay_samples < len(modified_audio):
                    reverb[delay_samples:] += modified_audio[:-delay_samples] * (0.2 * np.exp(-delay))
            
            modified_audio = modified_audio + reverb
            modified_audio = np.clip(modified_audio, -1, 1)

        elif effect == "quantum":
            # Quantum effect: Random phase shifts and probability-based effects
            t = np.arange(len(modified_audio)) / self.sample_rate
            
            # Create quantum uncertainty in the signal
            uncertainty = np.random.random(len(modified_audio)) > 0.5
            phase_shifts = np.random.uniform(0, 2*np.pi, len(modified_audio))
            
            # Apply random phase shifts
            modified_audio = modified_audio * np.exp(1j * phase_shifts * uncertainty)
            modified_audio = np.real(modified_audio)
            
            # Add quantum tunneling effect (sudden appearances/disappearances)
            tunnel_prob = 0.05
            tunnel_mask = np.random.random(len(modified_audio)) > tunnel_prob
            modified_audio = modified_audio * tunnel_mask
            
            # Add superposition of frequencies
            superposition = np.zeros_like(modified_audio)
            frequencies = [440, 880, 1320]
            for freq in frequencies:
                phase = np.random.random() * 2 * np.pi
                superposition += 0.1 * np.sin(2 * np.pi * freq * t + phase)
            
            # Combine with original signal
            modified_audio = modified_audio + superposition
            
            # Add quantum entanglement (delayed self-correlation)
            entangle = np.zeros_like(modified_audio)
            delay_samples = int(0.05 * self.sample_rate)
            if delay_samples < len(modified_audio):
                entangle[delay_samples:] = modified_audio[:-delay_samples] * 0.3
            
            modified_audio = modified_audio + entangle
            modified_audio = np.clip(modified_audio, -1, 1)

        # Apply common effects based on sliders
        if self.effect_params['reverb'] > 0:
            delay_samples = int(0.05 * self.sample_rate)
            reverb = np.zeros_like(modified_audio)
            reverb[delay_samples:] = modified_audio[:-delay_samples] * self.effect_params['reverb']
            modified_audio += reverb
            
        if self.effect_params['distortion'] > 0:
            modified_audio = np.clip(
                modified_audio * (1 + self.effect_params['distortion'] * 2),
                -1, 1
            )
            
        if self.effect_params['delay'] > 0:
            delay_samples = int(self.effect_params['delay'] * 0.5 * self.sample_rate)
            delay = np.zeros_like(modified_audio)
            delay[delay_samples:] = modified_audio[:-delay_samples] * 0.5
            modified_audio += delay
            
        if self.effect_params['flanger'] > 0:
            t = np.arange(len(modified_audio)) / self.sample_rate
            mod = np.sin(2 * np.pi * self.effect_params['flanger'] * t)
            modified_audio = modified_audio * (1 + 0.3 * mod)
            
        return modified_audio
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    voice_changer = VoiceChanger()
    voice_changer.run()
