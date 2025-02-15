const mongoose=require('mongoose')

const testimonial=mongoose.Schema({
    user:{type:mongoose.Schema.Types.ObjectId,ref:'User'},
    // email:{type:String,required:true,minLength:5},
    comment:{type:String,required:true}
})


const Testimonial=mongoose.model('Testimonial',testimonial);
module.exports=Testimonial