from playwright.sync_api import sync_playwright, expect


# Expensify app URL
EXPENSIFY_URL = "https://dev.new.expensify.com:8082/"

# Email login credentials
EMAIL_USERNAME = "naturesv057+52393_1@gmail.com"

def launch_browser(pw, device=None, geolocation=None):
    """
    Launch the browser.
    """
    browser = pw.chromium.launch(
        headless=False,
        proxy = {"server": "http://127.0.0.1:8080/"},
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=500
    )
    context_args = {}
    if device:
        context_args.update(pw.devices[device])
    if geolocation:
        context_args["geolocation"] = geolocation
        context_args["permissions"] = ["geolocation"]
    context = browser.new_context(**context_args)
    page = context.new_page()
    return browser, context, page


def login_user(page):
    """
    Login to the Expensify app and complete the onboarding.
    """
    # Open the Expensify app
    page.goto(EXPENSIFY_URL)

    # Login user
    page.get_by_role("textbox", name="Phone or email").fill(EMAIL_USERNAME)
    page.locator("div", has_text="Continue").last.click()

    # Get the OTP code from the email
    otp = "123456"
    
    # Enter the OTP code
    page.keyboard.type(otp)


def test_52393():
    with sync_playwright() as p:
        # Step 1: Login user
        browser, context, page = launch_browser(p)
        login_user(page)
        page.wait_for_timeout(5000)

        # Step 2: Reintroduce scenario
        page.get_by_label("Start chat (Floating action)").click()
        page.get_by_label("Start chat", exact=True).click()
        chat_email = "naturesv057+52393_1895457834@gmail.com"
        page.get_by_test_id("selection-list-text-input").fill(chat_email)
        page.get_by_label(chat_email).click()
        
        #Step 3: Check if the issue is resolved
        expect(page.get_by_label("Chat welcome message").last).to_contain_text("Use the + button to split, create, or pay an expense.")
