import * as ort from "onnxruntime-node";
import { Buffer } from "node:buffer";
import sharp from "sharp";

const modelName  = "model.onnx"
const inputPath  = "./inputImgs"
const outputPath = "./outputImgs"

export type YOLOBox = {
    x: number; // center x coordinate (pixels)
    y: number; // center y coordinate (pixels)
    w: number; // width of the box (pixels)
    h: number; // height of the box (pixels)
    conf: number; // confidence score (0-1)
};

console.log("Creating session...")
const session = await ort.InferenceSession.create(
    `./models/${modelName}`
);

console.log("Session inputMetaData:", session.inputMetadata);
// example inspecting the model input metadata
// [
//     {
//         name: "images",  // input tensor name
//         isTensor: true,  // confirm its a tensor
//         type: "float32", // data type of tensor
//         shape: [         // tensor dimensions
//             1,           // batch size
//             3,           // channels (array in ...R...G...B)
//             1280,        // height
//             1280         // width
//         ],
//     },
// ];

// get input data for later Tensor creation
const inputMeta = Object.values(session.inputMetadata)[0];
const [, , height, width] = inputMeta.shape;

const imgs = await getFiles(inputPath);

for (const img of imgs) {
    console.log(`Working on image: ${inputPath}/${img} ...`)
    const inputTensor = await embedImage(`${inputPath}/${img}`, height, width);

    const results = await session.run({ images: inputTensor });
    
    // console.log("### output ###");
    console.log(results.output0);
    // console.log(results.output0.dims);

    const rawBoxes = getBoxes(results.output0);
    // console.log("first box found:", rawBoxes[0]);
    const boxes = nms(rawBoxes);
    await drawBoxesOnImage(`${inputPath}/${img}`, `${outputPath}/${img}`, boxes, width, height);
};

// --- HELPER FUNCTIONS ---

async function getFiles(dirPath: string): Promise<string[]> {
    const files: string[] = [];

    try {
        for await (const dirEntry of Deno.readDir(dirPath)) {
            if (dirEntry.name == ".gitkeep") continue
            if (dirEntry.isFile) {
                files.push(`${dirEntry.name}`);
            }
        }
    } catch (error) {
        console.error(`Error reading directory: ${error.message}`);
        throw error;
    }

    return files;
}

/**
 * prepares an image for YOLOv8 inference.
 * 
 * @param image  - Path to the image file
 * @param height - Target model input height
 * @param width  - Target model input width
 * @returns An ONNX Runtime tensor ready for inference
 */
async function embedImage(image: string, height: number, width: number) {
    const pixels: Buffer<ArrayBufferLike> = await sharp(image)
        .resize({ width: width, height: height, fit: "contain" })
        .removeAlpha()
        .raw()
        .toBuffer();

    const red: number[] = [],
          green: number[] = [],
          blue: number[] = [];
    for (let i = 0; i < pixels.length; i += 3) {
        // normalize each channel value from [0, 255] → [0, 1]
        red.push(pixels[i] / 255.0);
        green.push(pixels[i + 1] / 255.0);
        blue.push(pixels[i + 2] / 255.0);
    }

    // flatten channels into expected format
    const prepared_input = [...red, ...green, ...blue];
    // convert to Float32Array
    const float32Data = Float32Array.from(prepared_input); 
    // Create ONNX Tensor with shape [batch, channels, height, width]
    const tensor = new ort.Tensor("float32", float32Data, [1, 3, height, width]);

    return tensor;
}

/**
 * Extract bounding boxes from YOLOv8 model output.
 *
 * This implementation is simplified for single-object detection
 * (e.g. detect one person, one banana, etc.).
 * 
 * @param output0   - The raw output tensor from ONNX Runtime
 * @param threshold - Confidence threshold (default: 0.6)
 * @returns An array of YOLOBox objects (x, y, w, h, conf)
 */
function getBoxes(output0: ort.Tensor, threshold: number = 0.6) {
    const data = output0.data;
    const boxes: YOLOBox[] = [];

    // YOLOv8 tensor layout (per prediction):
    // [x, y, w, h, confidence]

    const numBoxes = output0.dims[2]; // number of predicted boxes (model-specific)
    // tensor is in ...x...y...w...h...conf
    for (let i = 0; i < numBoxes; i++) {
        const x = Number(data[i]); // x channel
        const y = Number(data[i + numBoxes]); // y channel
        const w = Number(data[i + 2 * numBoxes]); // w channel
        const h = Number(data[i + 3 * numBoxes]); // h channel
        const conf = Number(data[i + 4 * numBoxes]); // conf channel

        if (conf >= threshold) {
            boxes.push({ x, y, w, h, conf });
        }
    }

    return boxes;
}

/**
 * Non-Maximum Suppression (NMS)
 *
 * Removes overlapping bounding boxes by keeping only the ones
 * with the highest confidence. This prevents multiple boxes
 * being predicted around the same object.
 *
 * For single-object detection (YOLOv8, one class),
 * this will usually return either 0 or 1 box,
 * but NMS is still applied for correctness.
 *
 * @param boxes        - Array of YOLOBox predictions
 * @param iouThreshold - Overlap threshold for suppression (default: 0.5)
 * @returns Filtered array of YOLOBox objects
 */
function nms(boxes: YOLOBox[], iouThreshold = 0.5) {
    // Sort boxes by confidence in descending order
    boxes.sort((a, b) => b.conf - a.conf);
    const keep: YOLOBox[] = [];

    /**
     * Compute Intersection-over-Union (IoU) between two boxes.
     *
     * IoU = (area of overlap) / (area of union)
     * Range: [0, 1]
     * 0 -> no overlap
     * 1 -> perfect overlap
     */
    function iou(boxA: YOLOBox, boxB: YOLOBox) {
        // Coordinates of overlap region
        const x1 = Math.max(boxA.x - boxA.w / 2, boxB.x - boxB.w / 2);
        const y1 = Math.max(boxA.y - boxA.h / 2, boxB.y - boxB.h / 2);
        const x2 = Math.min(boxA.x + boxA.w / 2, boxB.x + boxB.w / 2);
        const y2 = Math.min(boxA.y + boxA.h / 2, boxB.y + boxB.h / 2);

        // Intersection area (clamped to 0 if no overlap)
        const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

        // Areas of each box
        const boxAArea = boxA.w * boxA.h;
        const boxBArea = boxB.w * boxB.h;

        // IoU = overlap / union
        return interArea / (boxAArea + boxBArea - interArea);
    }

    // iterate through boxes, keeping only those that don't overlap too much
    // keeping the one with highest confidence.
    for (const box of boxes) {
        if (!keep.some((b) => iou(b, box) > iouThreshold)) {
            keep.push(box);
        }
    }
    return keep;
}

/**
 * Draws bounding boxes on an image and saves the result.
 *
 * Uses Sharp to overlay an SVG containing <rect> elements
 * on top of the original image. Each YOLO box is converted
 * from (center-x, center-y, width, height) into top-left
 * coordinates for correct SVG rendering.
 *
 * @param inputPath   - Path to the input image
 * @param outputPath  - Path to save the output image with boxes
 * @param boxes       - Array of bounding boxes { x, y, w, h, conf }
 * @param imageWidth  - Width of the image in pixels
 * @param imageHeight - Height of the image in pixels
 */
async function drawBoxesOnImage(
    inputPath: string,
    outputPath: string,
    boxes: { x: number; y: number; w: number; h: number; conf: number }[],
    imageWidth: number,
    imageHeight: number
) {
    console.log("writing images with size:", width, height)

    // Generate the SVG string for all bounding boxes.
    const svgRects = boxes
        .map((box) => {
            // Convert center-point coordinates to top-left for SVG.
            const x = box.x - box.w / 2;
            const y = box.y - box.h / 2;
            return `<rect x="${x}" y="${y}" width="${box.w}" height="${box.h}" fill="none" stroke="red" stroke-width="2"/>`;
        })
        .join("");
    
    // wrap into full SVG
    const svgImage = `<svg width="${imageWidth}" height="${imageHeight}" xmlns="http://www.w3.org/2000/svg">${svgRects}</svg>`;
    const overlayBuffer = Buffer.from(svgImage);

    // overlay & save
    await sharp(inputPath)
        .resize({ width: imageWidth, height: imageHeight, fit: "contain" })
        .composite([{ input: overlayBuffer, blend: "over" }])
        .toFile(outputPath);

    console.log("Saved image with boxes to:", outputPath);
}