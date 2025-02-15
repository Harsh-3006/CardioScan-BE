const mongoose=require('mongoose')

const user=mongoose.Schema({
    username:{type:String,required:true, minLength:2},
    email:{type:String,required:true,minLength:5,unique:true},
    password:{type:String,required:true,minLength:5},
    profession: {
        type: String,
        enum: ["Doctor","Cardiologist", "Patient", "ECG Operator","Others"], // Allowed values
        required: true // Ensures the field is mandatory
    },
    gender: {
        type: String,
        enum: ["male","female"], // Allowed values
        required: true // Ensures the field is mandatory
    },
    profilePic:{
        type:String,
    }
})


const User=mongoose.model('User',user);
module.exports=User