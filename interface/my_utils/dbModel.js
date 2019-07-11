const Pool = require('pg').Pool
const pool = new Pool({
  user: 'saurabh',
  host: 'localhost',
  database: 'quantifai',
  password: 'saurabh',
  port: 5432,
});

const table_name = {
    PORTFOLIO_METRICS_TABLE : "portfolio_metrics",
    PAIRWISE_METRICS_TABLE : "pairwise_metrics"

}
const getMetricsPortfolio = function (req, response){
    var filters =req.body.filters;
    var filter_string ='';
    for(var i =0 ; i <filters.length; i++){
        filter_string+=filters[i]+',';
    }
    filter_string= filter_string.substring(0,filter_string.length-1);
    
    var query = 'SELECT '+filter_string +' FROM '+table_name.PORTFOLIO_METRICS_TABLE; 
    console.log("Going to execute query:  ", query);

    pool.query(query, (error, results) => {
        if (error) {
            response.status(411).json({"message":"Invalid filters"});
        }
        else{
        // console.log("results",results,results.rows);
        response.status(200).json(results.rows)
        }
        
      });
}
const getMetricsPair = function( req, response){
    var filters =req.body.filters;
    var filter_string ='';
    for(var i =0 ; i <filters.length; i++){
        filter_string+=filters[i]+',';
    }
    filter_string= filter_string.substring(0,filter_string.length-1);
    
    var pair_1 = req.body.stock_1+' '+req.body.stock_2;
    var pair_2 = req.body.stock_2+' '+req.body.stock_1;
    var query = "SELECT "+filter_string +" FROM "+table_name.PAIRWISE_METRICS_TABLE +
    " WHERE index IN ('"+pair_1+"','"+pair_2+"')" ; 
    console.log("Going to execute query:  ", query);

    pool.query(query, (error, results) => {
        if (error) {
            response.status(411).json({"message":"Invalid filters"});
        }
        else{
        // console.log("results",results,results.rows);
        response.status(200).json(results.rows)
        }
        
      });
}

module.exports = {
    getMetricsPortfolio,
    getMetricsPair
}