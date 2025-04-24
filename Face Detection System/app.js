class FaceDetectionSystem {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.startBtn = document.getElementById('startBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.recognizeBtn = document.getElementById('recognizeBtn');
        this.emotionIndicator = document.getElementById('emotionIndicator');
        this.personNameIndicator = document.getElementById('personNameIndicator');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.ageValue = document.getElementById('ageValue');
        this.knownFaces = new Map();
        
        this.initializeButtons();
        this.loadModel();
    }

    async loadModel() {
        try {
            this.model = await blazeface.load();
            console.log('Face detection model loaded');
            this.showSuccess('Face detection model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
            this.showError('Failed to load face detection model');
        }
    }

    initializeButtons() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.captureBtn.addEventListener('click', () => this.captureFace());
        this.recognizeBtn.addEventListener('click', () => this.toggleRecognition());
        document.getElementById('addFaceBtn').addEventListener('click', () => this.addKnownFace());
    }

    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640,
                    height: 480,
                    facingMode: 'user'
                } 
            });
            this.video.srcObject = stream;
            this.startBtn.disabled = true;
            this.captureBtn.disabled = false;
            this.recognizeBtn.disabled = false;
            this.showSuccess('Camera started successfully');
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showError('Cannot access camera. Please check permissions.');
        }
    }

    async captureFace() {
        if (!this.model) {
            this.showError('Face detection model not loaded');
            return;
        }

        this.ctx.drawImage(this.video, 0, 0, 640, 480);
        const predictions = await this.model.estimateFaces(this.canvas);
        
        if (predictions.length > 0) {
            this.analyzeFace(predictions[0]);
            this.showSuccess('Face captured successfully');
        } else {
            this.showError('No face detected');
        }
    }

    async analyzeFace(face) {
        // Update confidence
        const confidence = (face.probability[0] * 100).toFixed(1);
        this.confidenceValue.textContent = `${confidence}%`;

        // Enhanced age estimation based on multiple facial features
        const faceWidth = face.bottomRight[0] - face.topLeft[0];
        const faceHeight = face.bottomRight[1] - face.topLeft[1];
        const faceRatio = faceWidth / faceHeight;
        
        // Calculate eye distance and position
        const eyeDistance = Math.abs(face.landmarks[1][0] - face.landmarks[0][0]);
        const eyeToNoseRatio = Math.abs(face.landmarks[2][1] - face.landmarks[0][1]) / faceHeight;
        
        // Calculate mouth size and position
        const mouthWidth = Math.abs(face.landmarks[3][0] - face.landmarks[4][0]);
        const mouthToNoseRatio = Math.abs(face.landmarks[3][1] - face.landmarks[2][1]) / faceHeight;

        // Advanced age estimation
        let estimatedAge = this.calculateAgeFromFeatures(
            faceRatio,
            eyeDistance / faceWidth,
            eyeToNoseRatio,
            mouthWidth / faceWidth,
            mouthToNoseRatio
        );
        this.ageValue.textContent = estimatedAge;

        // Analyze facial features
        const features = this.analyzeFacialFeatures(face);
        this.updateFeaturesList(features);

        // Update person recognition with improved accuracy
        const recognizedPerson = await this.recognizePerson(face);
        const personNameSpan = this.personNameIndicator.querySelector('span');
        if (recognizedPerson) {
            personNameSpan.textContent = recognizedPerson;
            personNameSpan.style.color = 'var(--success-color)';
        } else {
            personNameSpan.textContent = 'Unknown';
            personNameSpan.style.color = 'var(--text-light)';
        }

        // Enhanced emotion analysis
        const emotion = this.analyzeEmotion(face);
        this.emotionIndicator.querySelector('span').textContent = emotion.dominant;
        this.updateEmotionChart(emotion.probabilities);
    }

    calculateAgeFromFeatures(faceRatio, eyeRatio, eyeToNoseRatio, mouthRatio, mouthToNoseRatio) {
        let baseAge = 30;

        // Facial proportions adjustments
        if (faceRatio > 0.85) baseAge -= 5;
        if (faceRatio < 0.75) baseAge += 5;

        // Eye-based adjustments
        if (eyeRatio > 0.3) baseAge -= 3;
        if (eyeRatio < 0.25) baseAge += 3;

        // Facial feature positioning adjustments
        if (eyeToNoseRatio > 0.4) baseAge -= 4;
        if (eyeToNoseRatio < 0.35) baseAge += 4;

        // Mouth feature adjustments
        if (mouthRatio > 0.5) baseAge -= 2;
        if (mouthToNoseRatio > 0.6) baseAge -= 3;

        // Add natural variation
        baseAge += Math.random() * 4 - 2;

        // Ensure realistic age range
        return Math.max(15, Math.min(75, Math.round(baseAge)));
    }

    analyzeFacialFeatures(face) {
        const features = [];
        
        const eyeDistance = Math.abs(face.landmarks[1][0] - face.landmarks[0][0]);
        const faceWidth = face.bottomRight[0] - face.topLeft[0];
        const faceHeight = face.bottomRight[1] - face.topLeft[1];
        const eyeRatio = eyeDistance / faceWidth;
        const nosePosition = face.landmarks[2][1];
        const mouthWidth = Math.abs(face.landmarks[3][0] - face.landmarks[4][0]);

        features.push(`Eye Distance: ${eyeRatio.toFixed(2)}`);
        features.push(`Face Width: ${faceWidth.toFixed(0)}px`);
        features.push(`Face Height: ${faceHeight.toFixed(0)}px`);
        features.push(`Nose Position: ${nosePosition.toFixed(0)}px`);
        features.push(`Mouth Width: ${mouthWidth.toFixed(0)}px`);

        return features;
    }

    updateFeaturesList(features) {
        const featuresList = document.getElementById('featuresList');
        featuresList.innerHTML = features.map(feature => `<li>${feature}</li>`).join('');
    }

    async recognizePerson(face) {
        if (this.knownFaces.size === 0) return null;

        const faceData = await this.getFaceEmbedding(face);
        let bestMatch = null;
        let highestSimilarity = 0;

        for (const [name, knownFaceData] of this.knownFaces.entries()) {
            const similarity = await this.compareFaces(faceData, knownFaceData);
            if (similarity > highestSimilarity && similarity > 0.75) {
                highestSimilarity = similarity;
                bestMatch = name;
            }
        }

        return bestMatch;
    }

    async getFaceEmbedding(face) {
        const faceCanvas = document.createElement('canvas');
        const ctx = faceCanvas.getContext('2d');
        const [x, y] = face.topLeft;
        const [width, height] = [
            face.bottomRight[0] - face.topLeft[0],
            face.bottomRight[1] - face.topLeft[1]
        ];
        
        faceCanvas.width = width;
        faceCanvas.height = height;
        ctx.drawImage(this.canvas, x, y, width, height, 0, 0, width, height);
        
        return faceCanvas.toDataURL();
    }

    async compareFaces(face1Data, face2Data) {
        return new Promise((resolve) => {
            const img1 = new Image();
            const img2 = new Image();
            
            let loadedImages = 0;
            const onImageLoad = () => {
                loadedImages++;
                if (loadedImages === 2) {
                    const canvas1 = document.createElement('canvas');
                    const canvas2 = document.createElement('canvas');
                    const ctx1 = canvas1.getContext('2d');
                    const ctx2 = canvas2.getContext('2d');

                    const size = 100;
                    canvas1.width = canvas2.width = size;
                    canvas1.height = canvas2.height = size;

                    ctx1.drawImage(img1, 0, 0, size, size);
                    ctx2.drawImage(img2, 0, 0, size, size);

                    const data1 = ctx1.getImageData(0, 0, size, size).data;
                    const data2 = ctx2.getImageData(0, 0, size, size).data;

                    let similarity = 0;
                    for (let i = 0; i < data1.length; i += 4) {
                        const diff = Math.abs(data1[i] - data2[i]) +
                                   Math.abs(data1[i + 1] - data2[i + 1]) +
                                   Math.abs(data1[i + 2] - data2[i + 2]);
                        similarity += 1 - (diff / 765);
                    }

                    resolve(similarity / (data1.length / 4));
                }
            };

            img1.onload = onImageLoad;
            img2.onload = onImageLoad;
            img1.src = face1Data;
            img2.src = face2Data;
        });
    }

    analyzeEmotion(face) {
        const mouthWidth = Math.abs(face.landmarks[3][0] - face.landmarks[4][0]);
        const mouthHeight = Math.abs(face.landmarks[3][1] - face.landmarks[4][1]);
        const eyeDistance = Math.abs(face.landmarks[1][0] - face.landmarks[0][0]);
        
        const probabilities = {
            Happy: this.calculateHappiness(mouthWidth, mouthHeight),
            Sad: this.calculateSadness(mouthWidth, mouthHeight),
            Surprised: this.calculateSurprise(eyeDistance),
            Neutral: 0.2
        };

        const total = Object.values(probabilities).reduce((a, b) => a + b, 0);
        Object.keys(probabilities).forEach(key => {
            probabilities[key] = probabilities[key] / total;
        });

        return {
            dominant: Object.entries(probabilities).reduce((a, b) => a[1] > b[1] ? a : b)[0],
            probabilities
        };
    }

    calculateHappiness(mouthWidth, mouthHeight) {
        return mouthWidth / mouthHeight > 2 ? 0.6 : 0.2;
    }

    calculateSadness(mouthWidth, mouthHeight) {
        return mouthWidth / mouthHeight < 1.5 ? 0.5 : 0.2;
    }

    calculateSurprise(eyeDistance) {
        return eyeDistance > 100 ? 0.7 : 0.2;
    }

    updateEmotionChart(emotions) {
        const data = [{
            type: 'bar',
            x: Object.keys(emotions),
            y: Object.values(emotions),
            marker: {
                color: 'rgba(0, 255, 136, 0.6)'
            }
        }];

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6f1ff' },
            margin: { l: 40, r: 20, t: 20, b: 40 }
        };

        Plotly.newPlot('emotionChart', data, layout);
    }

    addKnownFace() {
        const nameInput = document.getElementById('personName');
        const name = nameInput.value.trim();
        
        if (!name) {
            this.showError('Please enter a name');
            return;
        }

        this.ctx.drawImage(this.video, 0, 0, 640, 480);
        const faceData = this.canvas.toDataURL('image/jpeg');
        
        this.knownFaces.set(name, faceData);
        this.updateKnownFacesList();
        nameInput.value = '';
        this.showSuccess('Face added to database');
    }

    updateKnownFacesList() {
        const container = document.getElementById('knownFacesList');
        container.innerHTML = '';

        this.knownFaces.forEach((faceData, name) => {
            const faceElement = document.createElement('div');
            faceElement.className = 'known-face';
            faceElement.innerHTML = `
                <img src="${faceData}" alt="${name}">
                <span>${name}</span>
            `;
            container.appendChild(faceElement);
        });
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message error';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 3000);
    }

    showSuccess(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'error-message success';
        successDiv.textContent = message;
        document.body.appendChild(successDiv);
        setTimeout(() => successDiv.remove(), 3000);
    }

    toggleRecognition() {
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
            this.recognitionInterval = null;
            this.recognizeBtn.innerHTML = '<i class="fas fa-search"></i> Recognize Face';
            return;
        }

        this.recognizeBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Recognition';
        this.recognitionInterval = setInterval(() => this.captureFace(), 1000);
    }
}

// Initialize the system when DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    new FaceDetectionSystem();
});