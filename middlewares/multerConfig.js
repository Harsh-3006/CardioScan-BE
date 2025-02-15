// const multer = require('multer');
// const path = require('path');
// const fs = require('fs');

// // Set up file upload using Multer
// const storage = multer.diskStorage({
//     destination: function (req, file, cb) {
//         cb(null, path.join(__dirname, '../../UPLOADS')); // Destination folder for images
//     },
//     filename: function (req, file, cb) {
//         cb(null, Date.now() + path.extname(file.originalname)); // Filename as current timestamp
//     },
// });

// const upload = multer({ storage: storage });

// module.exports = upload




const multer = require('multer');
const { CloudinaryStorage } = require('multer-storage-cloudinary');
const cloudinary = require('cloudinary').v2;

// Configure Cloudinary
cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET
});

// Set up Multer storage for Cloudinary
const storage = new CloudinaryStorage({
    cloudinary: cloudinary,
    params: {
        folder: 'CardioScan', // Cloudinary folder name
        // format: async (req, file) => 'png', // File format (optional, can be dynamic)
        public_id: (req, file) => Date.now().toString(), // Unique file name
    },
});

const upload = multer({ storage: storage });

module.exports = upload;
