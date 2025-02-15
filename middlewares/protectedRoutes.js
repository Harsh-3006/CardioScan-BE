const jwt = require("jsonwebtoken");
const User = require("../models/UserSchema");

const protectedRoute = async (req, res, next) => {
    try {
        // Extract token from headers
        const token = await req.headers["auth"]; // Corrected from `req.headers('auth')`
        console.log("token",token)
        // Check if token is provided
        if (!token || token==null) {
            return res.status(401).json({ error: "Unauthorized! Please log in first." });
        }

        // Verify token
        const decoded = jwt.verify(token, "harsh");
        console.log("Decoded Token:", decoded);
        if(!decoded._id){
            return res.status(401).json({ error: "Unauthorized! Please log in first." });
        }
        const user=await User.findById(decoded._id)
        console.log("user",user)

        // Attach user data to request for later use
        req.user = user;

        // Proceed to next middleware or route handler
        next();
    } catch (error) {
        console.log("error",error)
        // return res.status(401).json({ error: "Invalid or expired token!" });
        return res.status(401).json({ error: "Login First" });
    }
};

module.exports = protectedRoute;
