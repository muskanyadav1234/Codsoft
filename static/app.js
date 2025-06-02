class AIVisionSystem {
  constructor() {
      this.video = document.getElementById('video');
      this.canvas = document.getElementById('canvas');
      this.context = this.canvas.getContext('2d');
      this.isProcessing = false;
      this.capturedImage = null;
      this.currentFacingMode = 'user';
      this.processingTimeout = null;
      
      document.querySelector('.capture-btn').addEventListener('click', () => this.captureImage());
      document.querySelector('.analyze-btn').addEventListener('click', () => this.analyzeImage());
      document.querySelector('.switch-camera-btn')?.addEventListener('click', () => this.switchCamera());
      
      this.updateSystemStatus('initializing');
  }

  async initialize() {
      try {
          await this.setupCamera();
          this.updateSystemStatus('active');
      } catch (error) {
          console.error('Camera initialization error:', error);
          this.showError('Camera access denied. Please enable camera access.');
          this.updateSystemStatus('error');
      }
  }

  async setupCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({
          video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: this.currentFacingMode
          }
      });
      this.video.srcObject = stream;
      await this.video.play();
      
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      
      this.video.style.display = 'block';
      this.canvas.style.display = 'none';
  }

  async switchCamera() {
      if (this.isProcessing) return;
      
      this.currentFacingMode = this.currentFacingMode === 'user' ? 'environment' : 'user';
      
      if (this.video.srcObject) {
          this.video.srcObject.getTracks().forEach(track => track.stop());
      }
      
      try {
          await this.setupCamera();
      } catch (error) {
          console.error('Camera switch error:', error);
          this.showError('Failed to switch camera');
      }
  }

  async captureImage() {
      if (!this.video.srcObject) {
          this.showError('Camera not initialized');
          return;
      }

      try {
          this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
          this.capturedImage = this.canvas.toDataURL('image/jpeg', 0.9);
          
          const captureBtn = document.querySelector('.capture-btn');
          captureBtn.innerHTML = '<i class="fas fa-check"></i> Captured!';
          
          this.canvas.style.display = 'block';
          this.video.style.display = 'none';
          
          setTimeout(() => {
              this.canvas.style.display = 'none';
              this.video.style.display = 'block';
              captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture';
          }, 1000);
          
          this.showSuccess('Image captured successfully');
      } catch (error) {
          console.error('Capture error:', error);
          this.showError('Failed to capture image');
      }
  }

  async analyzeImage() {
      if (!this.capturedImage) {
          this.showError('Please capture an image first');
          return;
      }
      
      if (this.isProcessing) return;
      this.isProcessing = true;
      
      this.showLoading();
      this.resetUI();
      
      this.processingTimeout = setTimeout(() => {
          this.hideLoading();
          this.showError('Processing timeout. Please try again.');
          this.isProcessing = false;
      }, 15000);

      try {
          const response = await fetch('/analyze', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({ image: this.capturedImage })
          });
          
          clearTimeout(this.processingTimeout);
          
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const results = await response.json();
          this.displayResults(results);
          this.updateConfidenceMeters(results);
          this.showSuccess('Analysis completed successfully');
      } catch (error) {
          console.error('Analysis error:', error);
          this.handleServerError(error);
      } finally {
          clearTimeout(this.processingTimeout);
          this.isProcessing = false;
          this.hideLoading();
      }
  }

  updateConfidenceMeters(results) {
      // Face Detection Meter
      const faceMeter = document.querySelector('#faceResults + .confidence-meter .meter-fill');
      const faceConfidence = results.faces_detected > 0 ? 95 : 0;
      if (faceMeter) {
          faceMeter.style.width = `${faceConfidence}%`;
          faceMeter.style.backgroundColor = faceConfidence > 70 ? '#00ff88' : '#ff4444';
      }

      // Object Recognition Meter
      const objectMeter = document.querySelector('#objectResults + .confidence-meter .meter-fill');
      if (objectMeter && results.classifications?.[0]) {
          const objConfidence = results.classifications[0].confidence * 100;
          objectMeter.style.width = `${objConfidence}%`;
          objectMeter.style.backgroundColor = objConfidence > 70 ? '#00ff88' : '#ff4444';
      }

      // AI Analysis Meter
      const aiMeter = document.querySelector('#analysisResults + .confidence-meter .meter-fill');
      if (aiMeter && results.classifications?.[0]) {
          const aiConfidence = results.classifications[0].confidence * 100;
          aiMeter.style.width = `${aiConfidence}%`;
          aiMeter.style.backgroundColor = aiConfidence > 70 ? '#00ff88' : '#ff4444';
      }
  }

  updateSystemStatus(status) {
      const statusDot = document.querySelector('.status-dot');
      const statusText = document.querySelector('.status-text');
      const systemStatus = document.querySelector('.system-status');
      
      if (statusDot && statusText) {
          statusDot.className = `status-dot status-${status}`;
          statusText.textContent = `System ${status.charAt(0).toUpperCase() + status.slice(1)}`;
      }
      
      if (systemStatus) {
          systemStatus.textContent = `System ${status.charAt(0).toUpperCase() + status.slice(1)}`;
      }
  }

  displayResults(results) {
      if (results.error) {
          this.showError(results.error);
          return;
      }

      const faceResults = document.getElementById('faceResults');
      faceResults.innerHTML = `
          <div class="result-item">
              <p>Faces detected: ${results.faces_detected}</p>
              <p class="confidence">Confidence: ${results.faces_detected > 0 ? '95' : '0'}%</p>
          </div>
      `;

      const objectResults = document.getElementById('objectResults');
      objectResults.innerHTML = results.classifications.map(item => `
          <div class="result-item">
              <p class="label">${item.label}</p>
              <p class="confidence">${(item.confidence * 100).toFixed(1)}%</p>
          </div>
      `).join('');

      const analysisResults = document.getElementById('analysisResults');
      const overallConfidence = results.classifications[0]?.confidence || 0;
      analysisResults.innerHTML = `
          <div class="result-item">
              <p>Analysis Complete</p>
              <p class="confidence">Overall Confidence: ${(overallConfidence * 100).toFixed(1)}%</p>
          </div>
      `;
  }

  showLoading() {
      document.getElementById('loading-overlay').classList.remove('hidden');
      ['faceResults', 'objectResults', 'analysisResults'].forEach(id => {
          document.getElementById(id).innerHTML = '<div class="pulse-animation">Analyzing...</div>';
      });
  }

  hideLoading() {
      document.getElementById('loading-overlay').classList.add('hidden');
  }

  showError(message) {
      const errorDiv = document.createElement('div');
      errorDiv.className = 'error-message';
      errorDiv.textContent = message;
      document.querySelector('.camera-section').appendChild(errorDiv);
      setTimeout(() => errorDiv.remove(), 3000);
  }

  showSuccess(message) {
      const successDiv = document.createElement('div');
      successDiv.className = 'success-message';
      successDiv.textContent = message;
      document.querySelector('.camera-section').appendChild(successDiv);
      setTimeout(() => successDiv.remove(), 3000);
  }

  resetUI() {
      ['faceResults', 'objectResults', 'analysisResults'].forEach(id => {
          const element = document.getElementById(id);
          if (element) {
              element.innerHTML = '<div class="pulse-animation">Waiting for analysis...</div>';
          }
      });

      document.querySelectorAll('.confidence-meter .meter-fill').forEach(meter => {
          meter.style.width = '0%';
          meter.style.backgroundColor = '#666';
      });
  }

  handleServerError(error) {
      console.error('Server Error:', error);
      this.showError('Server error occurred. Please try again later.');
      this.resetUI();
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  const aiSystem = new AIVisionSystem();
  aiSystem.initialize();

  // Initialize particles.js
  particlesJS('particles-js', {
      particles: {
          number: { value: 80, density: { enable: true, value_area: 800 } },
          color: { value: "#00ff88" },
          shape: { type: "circle" },
          opacity: { value: 0.5, random: false },
          size: { value: 3, random: true },
          line_linked: {
              enable: true,
              distance: 150,
              color: "#00ccff",
              opacity: 0.4,
              width: 1
          },
          move: {
              enable: true,
              speed: 6,
              direction: "none",
              random: false,
              straight: false,
              out_mode: "out",
              bounce: false
          }
      },
      interactivity: {
          detect_on: "canvas",
          events: {
              onhover: { enable: true, mode: "repulse" },
              onclick: { enable: true, mode: "push" },
              resize: true
          }
      },
      retina_detect: true
  });
});