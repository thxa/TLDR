curl -X GET -H 'Accept:application/json' http://localhost:8080/route-plans/ID > plan.json
cat plan.json | jq > a.json

