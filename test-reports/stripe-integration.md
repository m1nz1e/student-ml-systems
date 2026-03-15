# Stripe Integration Report — QuikScore

**Date:** 2026-03-15  
**Status:** ✅ Implemented  
**Branch:** `main`  

## Summary

Full Stripe payment integration implemented for QuikScore with checkout sessions, webhooks, customer portal, and database sync.

## What Was Done

### 1. `payments.py` — Complete Rewrite
- **Checkout session creation** using real Stripe API (`stripe.checkout.Session.create`)
  - Validates plan (starter/pro/business)
  - Creates subscription-mode checkout with proper price IDs
  - Returns redirect URL for frontend
  - Enables promotion codes, requires billing address
- **Plan mapping via env vars:**
  - `STRIPE_PRICE_ID_STARTER` → £49/mo
  - `STRIPE_PRICE_ID_PRO` → £99/mo
  - `STRIPE_PRICE_ID_BUSINESS` → £299/mo

### 2. Webhook Endpoint (`POST /api/payments/webhook`)
- Signature verification via `STRIPE_WEBHOOK_SECRET`
- **5 event handlers:**

| Event | Action |
|-------|--------|
| `checkout.session.completed` | Activate subscription, upsert subscriptions table, update user |
| `customer.subscription.updated` | Sync plan changes, update tier & limits |
| `customer.subscription.deleted` | Downgrade to free, mark cancelled |
| `invoice.payment_succeeded` | Reset usage counters, extend period |
| `invoice.payment_failed` | Mark past_due → suspended after 3 attempts |

### 3. Customer Portal
- **`POST /api/payments/create-portal-session`** — accepts `customer_id` in body
- **`POST /api/payments/portal`** — authenticated, looks up customer ID from JWT
- Returns Stripe Billing Portal URL for self-service management

### 4. Database Integration
- **`users` table:** Updates `subscription_tier`, `subscription_status`, `stripe_customer_id`
- **`subscriptions` table:** Full upsert with `stripe_subscription_id`, `stripe_price_id`, tier, limits, period dates
- **Tier limits:**
  - Starter: 10 reports, 0 property searches
  - Pro: 100 reports, 50 property searches
  - Business: Unlimited

### 5. Additional Endpoints
- `GET /api/payments/stripe-publishable-key` — safe key exposure for frontend
- `GET /api/payments/config` — plans + publishable key in one call

## Files Modified

| File | Change |
|------|--------|
| `backend/payments.py` | Full rewrite with Stripe integration |
| `backend/services/stripe_service.py` | Already existed — kept as reference/service layer |

## Environment Variables Required (Render)

```
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_ID_STARTER=price_...
STRIPE_PRICE_ID_PRO=price_...
STRIPE_PRICE_ID_BUSINESS=price_...
```

## Testing

### Local/CLI test
```bash
curl -X POST https://quikscore.onrender.com/api/payments/create-checkout-session \
  -H "Content-Type: application/json" \
  -d '{
    "email":"test@test.com",
    "plan":"pro",
    "success_url":"https://quik-score.vercel.app/billing?success=true",
    "cancel_url":"https://quik-score.vercel.app/pricing"
  }'
```

### Webhook test (Stripe CLI)
```bash
stripe listen --forward-to https://quikscore.onrender.com/api/payments/webhook
stripe trigger checkout.session.completed
```

## Next Steps

1. **Create Stripe Products & Prices** in Stripe Dashboard (test mode) for the 3 plans
2. **Set env vars** in Render dashboard with real price IDs
3. **Configure webhook endpoint** in Stripe Dashboard pointing to `/api/payments/webhook`
4. **Test end-to-end** with Stripe test cards (4242 4242 4242 4242)
5. **Switch to live keys** when ready for production

## Notes

- Uses Stripe test mode keys first (`sk_test_` / `pk_test_`)
- No API keys committed to repo
- Frontend already has Stripe integration built in (`@stripe/react-stripe-js`)
- The existing `stripe_service.py` service class is kept as a reference but `payments.py` now handles everything directly
