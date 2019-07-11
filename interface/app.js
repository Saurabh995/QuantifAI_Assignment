const express = require('express');
const Api = require('./my_utils/api');
const dbModel = require('./my_utils/dbModel');
const app = express();
const port = 3000;
var bodyParser = require('body-parser')
app.use(bodyParser.urlencoded({ extended: false }))
// parse application/json
app.use(bodyParser.json());

app.get('/_status', (req, res) => res.send('Success'))
app.post('/portfolio/metrics',dbModel.getMetricsPortfolio);

app.post('/pairwise/metrics', dbModel.getMetricsPair);
app.listen(port, () => console.log(`Example app listening on port ${port}!`))