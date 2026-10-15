import json
from pathlib import Path
import numpy as np
import pytest
from main import parse_args


def test_end_to_end_artifacts(integration_run):
    run = integration_run
    out = run["root"] / "outputs/synthetic"
    for name in ['metrics.json','data_quality.json','model_comparison.csv','forecast.csv',
                 'forecast_168h.csv','backtest_predictions.csv','simulation.json','run_manifest.json']:
        assert (out/name).is_file()
    assert run['metrics']['selection_rule'].startswith('Minimum validation')
    assert run['forecast'].index.min() > run['frame'].index.max()
    assert run['simulation']['baseline_energy_kwh'] == pytest.approx(run['simulation']['shifted_energy_kwh'])
    assert (run['root']/'models/synthetic/forecast_bundle.joblib').exists()


def test_health_and_dashboard(integration_run):
    client = integration_run['client']
    assert client.get('/health').json['ready_runs'] == ['synthetic']
    assert client.get('/').status_code == 200
    assert b'No live grid connection' in client.get('/').data


def test_valid_forecast_post_and_get(integration_run):
    client = integration_run['client']
    post = client.post('/predict', json={'dataset':'synthetic','hours':24})
    assert post.status_code == 200
    value = post.json
    assert value['hours'] == 24 and len(value['predictions']) == 24
    get = client.get('/api/forecast?dataset=synthetic&hours=24')
    assert get.json['predictions'] == value['predictions']
    assert value['source_type'] == 'synthetic'


@pytest.mark.parametrize('body', [{'hours':0},{'hours':169},{'hours':True},{'hours':24.3},
 {'hours':'foo'},{'dataset':'../../etc'},{'dataset':[]},{'hour':14,'day':2},[],None])
def test_bad_predict_returns_400(integration_run, body):
    res = integration_run['client'].post('/predict', data=json.dumps(body), content_type='application/json')
    assert res.status_code == 400
    assert res.json['error'] == 'invalid_request'


def test_malformed_json_and_large_request(integration_run):
    client = integration_run['client']
    assert client.post('/predict', data='{bad', content_type='application/json').status_code == 400
    assert client.post('/predict', data=' '*17000, content_type='application/json').status_code == 413


def test_api_summary_history_and_evidence(integration_run):
    client = integration_run['client']
    for endpoint in ['/api/runs','/api/summary','/api/history','/api/evaluation']:
        res = client.get(endpoint)
        assert res.status_code == 200
        json.loads(res.data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def test_simulation_endpoint_and_validation(integration_run):
    client = integration_run['client']
    result = client.post('/api/simulate', json={'dataset':'synthetic','settings':{'flexible_fraction':.2}})
    assert result.status_code == 200
    r = result.json['result']
    assert r['baseline_energy_kwh'] == pytest.approx(r['shifted_energy_kwh'])
    assert client.post('/api/simulate',json={'settings':{'offpeak_rate':100}}).status_code == 400


def test_download_allowlist(integration_run):
    client = integration_run['client']
    res = client.get('/download/synthetic/metrics.json')
    assert res.status_code == 200
    assert 'attachment' in res.headers['Content-Disposition']
    assert client.get('/download/synthetic/forecast_bundle.joblib').status_code == 400
    assert client.get('/images/synthetic/unknown.png').status_code == 400


@pytest.mark.parametrize('args', [['--horizon','0'],['--run-id','../../secret'],['--run-id','']])
def test_cli_input_validation(args):
    with pytest.raises(SystemExit):
        parse_args(args)
