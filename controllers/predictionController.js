const { PythonShell } = require('python-shell');
const path = require('path');

const predict = (req, res) => {
    console.log("Request received for prediction...");
    const imagePath = req.file.path; // Path to the uploaded image
    const modelPath = process.env.MODEL_PATH;

    console.log("Image Path:", imagePath);
    console.log("Model Path:", modelPath);

    const options = {
        scriptPath: path.join(__dirname, '../scripts'),
        args: [imagePath, modelPath],
        pythonOptions: ['-u'], // Ensure unbuffered output
    };

    console.log("Going to run PythonShell...");
    PythonShell.run('predict.py', options, (err, results) => {
        if (err) {
            console.error("Error executing Python script:", err.message);
            return res.status(500).json({
                error: "Internal Server Error",
                details: err.message,
            });
        }

        console.log("Python script executed successfully.");
        console.log("Results:", results);

        if (results && results.length >= 2) {
            const predictedClass = results[0].trim();
            const confidence = results[1].trim();
            return res.json({
                prediction: predictedClass === '1' ? 'Abnormal' : 'Normal',
                confidence: `${confidence}%`,
            });
        }

        return res.status(500).json({ error: "Unexpected error in Python script output" });
    });
};

module.exports = { predict };
