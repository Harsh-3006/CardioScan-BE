const express = require("express");
const User = require("../models/UserSchema.js"); // Import the Mongoose model
// const userValidateRequest = require("../middlewares/validation/userValidation.js"); // will see later
const { userSignupSchema, userLoginSchema } = require('../validation/userValidation.js')
const jwt = require('jsonwebtoken')

const router = express.Router();

router.post("/signup", async (req, res) => {
    const { error } = userSignupSchema.validate(req.body);
    if (error) return res.status(400).json({
        error: error.details[0].message

    });
    const { username, email, password, profession,gender } = req.body;

    try {

        const checkUser = await User.findOne({ email })
        if (checkUser) {
            return res.status(401).json({ error: "User already exist" })
        }
        const boyProfilePic=`https://avatar.iran.liara.run/public/boy?username=${username}`
        const girlProfilePic=`https://avatar.iran.liara.run/public/girl?username=${username}`
        // Create a new user instance
        const newUser = new User({ username, email, password, profession,
            profilePic:gender==='male'?boyProfilePic:girlProfilePic,
            gender
         });

        // Save the user to MongoDB
        await newUser.save();

        const user = {
            _id: newUser._id.toString() // Ensure it's a string
        };

        // Generate a token

        const token = jwt.sign(user, "harsh");

        res.status(201).json({ message: "User registered successfully!", token });
    } catch (error) {
        res.status(500).json({ error: "Something went wrong!" });
    }
});

router.post("/login", async (req, res) => {

    const { error } = userLoginSchema.validate(req.body);
    if (error) return res.status(400).json({
        error: error.details[0].message

    });

    const { email, password } = req.body;

    try {
        // Create a new user instance
        console.log(email,password)
        const checkUser = await User.findOne({ email })
        // console.log("checkuser",checkUser)
        if(!checkUser){
            return res.status(400).json({ error: "User not found" })
        }
        if (checkUser.password != password) {
            return res.status(400).json({ error: "Email or password not correct" })
        }

        const user = {
            _id: checkUser._id.toString() // Ensure it's a string
        };

        // Generate a token

        const token = jwt.sign(user, "harsh");

        res.status(201).json({ message: "User logged in successfully!", token });
    } catch (error) {
        console.log("error",error)
        res.status(500).json({ error: "Something went wrong!" });
    }
});

module.exports = router;
