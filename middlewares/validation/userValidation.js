const {userValidationSchema} = require("../validation/userValidation");

const userValidateRequest = (req, res, next) => {
    const { error } = userValidationSchema.validate(req.body, { abortEarly: false });
    if (error) {
        return res.status(400).json({ error: error.details[0].message });
    }
    next(); // Proceed to the next middleware or route handler
};

module.exports = userValidateRequest;
