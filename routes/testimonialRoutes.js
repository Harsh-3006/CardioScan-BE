const express = require("express");
const User = require("../models/UserSchema.js");
const Testimonial=require('../models/TestimonialSchema.js') // Import the Mongoose model
// const userValidateRequest = require("../middlewares/validation/userValidation.js"); // will see later
const { addTestimonial } = require('../validation/testimonialValidation.js')
// const jwt = require('jsonwebtoken')
const protectedRoute=require('../middlewares/protectedRoutes.js')

const router = express.Router();

router.post("/",protectedRoute, async (req, res) => {
    const { error } = addTestimonial.validate(req.body);
    if (error) return res.status(400).json({
        error: error.details[0].message
    });
    const { comment } = req.body;
    try {
        const _id = req.user._id
        const checkUser = await User.findById(_id)
        if (!checkUser) {
            return res.status(401).json({ error: "User did not exist please login first" })
        }
        // Create a new user instance
        const newTestimonial = new Testimonial({user:_id ,comment:comment.trim() });

        // Save the user to MongoDB
        await newTestimonial.save();

        res.status(201).json({ message: "Feedback saved", });
    } catch (error) {
        console.log(error)
        res.status(500).json({ error: "Something went wrong!" });
    }
});

router.get("/", async (req, res) => {

    try {
        // Create a new user instance
        const getTestinmonials = await Testimonial.find().populate('user')

        res.status(201).json({ message: "Displaying all testimonials", getTestinmonials });
    } catch (error) {
        res.status(500).json({ error: "Something went wrong!" });
    }
});

module.exports = router;
