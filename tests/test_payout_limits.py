import pytest
from unittest.mock import AsyncMock
from api.deriv_client import DerivClient


@pytest.mark.asyncio
async def test_get_contract_proposal_auto_scaling() -> None:
    """Test that get_contract_proposal auto-scales requested stake on payout limits."""
    # Instantiate client with a dummy access token
    client = DerivClient(access_token="test_token")

    # Mock the WebSocket send and receive method
    mock_send = AsyncMock()
    client._send_and_receive_trading = mock_send

    # Configure side effects for send_and_receive
    # First response: payout limit error
    error_response = {
        "error": {
            "message": "Minimum stake of 0.50 and maximum payout of 100.00. Current payout is 130.71.",
            "code": "PayoutLimitExceeded",
        }
    }

    # Second response: successful proposal with scaled down stake
    success_response = {
        "proposal": {
            "id": "prop_98765",
            "ask_price": 7.57,
            "payout": 99.00,
        }
    }

    mock_send.side_effect = [error_response, success_response]

    # Call get_contract_proposal with initial stake of 10.0
    proposal = await client.get_contract_proposal(
        symbol="frxEURGBP",
        contract_type="PUT",
        amount=10.0,
        duration=15,
        duration_unit="m",
    )

    # Verify final returned proposal is the successful scaled-down one
    assert "id" in proposal
    assert proposal["id"] == "prop_98765"
    assert proposal["ask_price"] == 7.57

    # Verify it retried exactly once
    assert mock_send.call_count == 2

    # Verify first request sent the original amount of 10.0
    first_call_req = mock_send.call_args_list[0][0][0]
    assert first_call_req["amount"] == 10.0

    # Verify second request sent the adjusted amount
    second_call_req = mock_send.call_args_list[1][0][0]
    assert second_call_req["amount"] < 10.0
    # Expected: 10.0 * (100.00 / 130.71) * 0.99 = 7.5739...
    assert abs(second_call_req["amount"] - (10.0 * (100.00 / 130.71) * 0.99)) < 0.01


@pytest.mark.asyncio
async def test_get_contract_proposal_rounding() -> None:
    """Test that get_contract_proposal rounds requested stake to 2 decimal places."""
    client = DerivClient(access_token="test_token")
    mock_send = AsyncMock(return_value={"proposal": {"id": "prop_12345", "ask_price": 5.0}})
    client._send_and_receive_trading = mock_send

    proposal = await client.get_contract_proposal(
        symbol="frxEURGBP",
        contract_type="PUT",
        amount=10.123456,
        duration=15,
        duration_unit="m",
    )

    assert "id" in proposal
    assert proposal["id"] == "prop_12345"

    # Verify amount was rounded to 2 decimal places in the request
    sent_req = mock_send.call_args_list[0][0][0]
    assert sent_req["amount"] == 10.12

