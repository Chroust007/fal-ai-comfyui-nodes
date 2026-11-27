import { app } from "/scripts/app.js";

// Přidává živý náhled kamery a tlačítko pro pořízení snímku k uzlu FAL_WebcamCapture.
// Snímek se ukládá jako base64 data URL přímo do skrytého STRING widgetu "image" -
// žádný soubor se nikam neukládá, vše zůstává jen v paměti prohlížeče/pythonu.
app.registerExtension({
    name: "fal.WebcamCapture",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FAL_WebcamCapture") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const imageWidget = node.widgets.find((w) => w.name === "image");
            const widthWidget = node.widgets.find((w) => w.name === "width");
            const heightWidget = node.widgets.find((w) => w.name === "height");
            const captureOnQueueWidget = node.widgets.find((w) => w.name === "capture_on_queue");

            if (imageWidget) {
                imageWidget.type = "hidden";
                imageWidget.computeSize = () => [0, 0];
            }

            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.gap = "4px";
            container.style.width = "100%";

            const video = document.createElement("video");
            video.style.width = "100%";
            video.autoplay = true;
            video.muted = true;
            video.playsInline = true;
            container.appendChild(video);

            const canvas = document.createElement("canvas");
            canvas.style.display = "none";
            container.appendChild(canvas);

            const button = document.createElement("button");
            button.textContent = "Pořídit snímek";
            button.style.width = "100%";
            container.appendChild(button);

            let stream = null;
            let autoCaptureTimer = null;

            const captureFrame = () => {
                if (!video.videoWidth || !video.videoHeight) {
                    return;
                }
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                if (imageWidget) {
                    imageWidget.value = canvas.toDataURL("image/png");
                }
            };

            const stopAutoCapture = () => {
                if (autoCaptureTimer) {
                    clearInterval(autoCaptureTimer);
                    autoCaptureTimer = null;
                }
            };

            const startAutoCapture = () => {
                stopAutoCapture();
                // Průběžně (2x za sekundu) drží widget "image" aktuální, aby byl snímek
                // připravený v okamžiku spuštění workflow (pokud je capture_on_queue zapnuté).
                autoCaptureTimer = setInterval(() => {
                    if (captureOnQueueWidget && captureOnQueueWidget.value) {
                        captureFrame();
                    }
                }, 500);
            };

            const startCamera = async () => {
                try {
                    const width = widthWidget?.value || 640;
                    const height = heightWidget?.value || 480;
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: { ideal: width }, height: { ideal: height } },
                        audio: false,
                    });
                    video.srcObject = stream;
                    startAutoCapture();
                } catch (err) {
                    console.error("FAL_WebcamCapture: nepodařilo se získat přístup ke kameře", err);
                }
            };

            button.addEventListener("click", captureFrame);

            node.addDOMWidget("webcam_preview", "preview", container, {
                serialize: false,
            });

            startCamera();

            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                stopAutoCapture();
                if (stream) {
                    stream.getTracks().forEach((track) => track.stop());
                }
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            return result;
        };
    },
});
