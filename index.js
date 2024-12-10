const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const admin = require('firebase-admin');
const sharp = require('sharp');
const cors = require('cors');

const serviceAccount = path.join(__dirname, 'service-account.json');
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});
const db = admin.firestore();


const app = express();
const port = process.env.PORT || 3000;

app.use(cors());

const upload = multer({ storage: multer.memoryStorage() });

async function loadModel() {
  const modelUrl = `https://storage.googleapis.com/bucket-windyc/submissions-model/model.json`;
  const model = await tf.loadGraphModel(modelUrl);
  console.log('Model loaded successfully');
  return model;
}

function preprocessImage(imageBuffer) {
  return tf.node.decodeImage(imageBuffer, 3)
    .resizeNearestNeighbor([224, 224])
    .expandDims()
    .toFloat();
}


app.post('/predict', upload.single('image'), async (req, res) => {
  // Validate file size
  if (req.file.size > 1000000) {
    return res.status(413).json({
      status: 'fail',
      message: 'Payload content length greater than maximum allowed: 1000000',
    });
  }

  // Validate metadata
  if ((await sharp(req.file.buffer).metadata()).channels === 1) {
    return res.status(400).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }

  try {
    const imageTensor = preprocessImage(req.file.buffer);
    const predictions = await model.predict(imageTensor).data();
    const result = predictions[0] > 0.5 ? 'Cancer' : 'Non-cancer';
    const id = uuidv4();

    await db.collection('predictions').doc(id).set({
      id: id,
      result: result,
      suggestion: result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.',
      createdAt: new Date().toISOString(),
    });

    res.status(201).json({
      status: 'success',
      message: 'Model is predicted successfully',
      data: {
        id: id,
        result: result,
        suggestion: result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.',
        createdAt: new Date().toISOString(),
      },
    });
  } catch (error) {
    console.error(error);
    res.status(400).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }
});

app.get('/predict/histories', async (req, res) => {
  try {
    const snapshot = await db.collection('predictions').get();
    const histories = [];

    snapshot.forEach(doc => {
      histories.push({
        id: doc.id,
        history: {
          result: doc.data().result,
          createdAt: doc.data().createdAt,
          suggestion: doc.data().suggestion,
          id: doc.data().id,
        },
      });
    });

    res.json({
      status: 'success',
      data: histories,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam mengambil riwayat prediksi',
    });
  }
});


let model;
loadModel().then(loadedModel => {
  model = loadedModel;
  app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
  });
}).catch(err => {
  console.error('Error loading model:', err);
});