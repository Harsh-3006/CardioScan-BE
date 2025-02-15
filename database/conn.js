const mongoose = require('mongoose')

const conn = () => {
    mongoose.connect(process.env.Mongo_URI)
        .then(() => {
            console.log('connected to mongoDb')
        }).catch((error) => {
            console.log("unable to connect", error)
        })
}

module.exports = conn