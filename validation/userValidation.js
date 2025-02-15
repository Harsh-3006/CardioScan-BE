const Joi = require("joi");

const userSignupSchema = Joi.object({
    username: Joi.string()
        .min(2)
        .required()
        .messages({
            "string.empty": "Username is required!",
            "string.min": "Username must be at least 2 characters long!"
        }),

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

    profession: Joi.string()
        .valid("Doctor", "Cardiologist", "Patient", "ECG Operator", "Others")
        .required()
        .messages({
            "any.only": "Profession must be either 'doctor', 'patient','Cardiologist' or 'others'!",
            "string.empty": "Profession is required!"
        }),

    gender: Joi.string()
        .valid("male","female")
        .required()
        .messages({
            "any.only": "Gender must be either 'Male', or 'Female'",
            "string.empty": "Gender is required!"
        })
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


module.exports = { userSignupSchema, userLoginSchema };
