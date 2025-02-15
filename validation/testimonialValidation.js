const Joi = require("joi");

const addTestimonial = Joi.object({

    comment: Joi.string()
        // .min(5)
        .required()
        .messages({
            "string.empty": "comment is required!",
            // "string.min": "Comment must be at least 5 characters long!"
        }),
});

const userLoginSchema = Joi.object({
    email: Joi.string()
        .email()
        .required()
        .messages({
            "string.empty": "Email is required!",
            "string.email": "Invalid email format! Please enter a valid email."
        }),

    password: Joi.string()
        .min(5)
        .required()
        .messages({
            "string.empty": "Password is required!",
            "string.min": "Password must be at least 5 characters long!"
        }),
});


module.exports = { addTestimonial };
